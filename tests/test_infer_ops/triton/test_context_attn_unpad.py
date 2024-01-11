import pytest
import torch
import torch.nn.functional as F
from packaging import version

from colossalai.kernel.triton import context_attention_unpadded
from colossalai.utils import get_current_device

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


def torch_attn_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int, num_heads: int, head_size: int):
    # For a single sequence, q,k,v [seq_len, num_heads, head_size]
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == head_size
    q = q.view(seq_len, num_heads, head_size)
    k = k.view(seq_len, num_heads, head_size)
    v = v.view(seq_len, num_heads, head_size)
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    mask = torch.tril(torch.ones(1, seq_len, seq_len), diagonal=0).to(device=get_current_device())
    mask[mask == 0.0] = float("-inf")
    mask = mask.repeat(num_heads, 1, 1)

    qk = torch.matmul(q, k.transpose(1, 2))
    attn_scores = qk / (head_size**0.5)
    attn_weights = F.softmax(attn_scores.to(dtype=torch.float32) + mask, dim=-1).to(dtype=q.dtype)
    out = torch.matmul(attn_weights, v).transpose(0, 1).contiguous()
    out = out.reshape(-1, num_heads, head_size)
    return out


def torch_attn_unpad(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context_lengths: torch.Tensor):
    # Process sequence one by one and cat them together.
    # q,k,v [num_tokens(sum(context_lengths)), num_heads, head_size]
    assert context_lengths.dim() == 1, "context_lengths should be a 1D tensor"
    _, num_heads, head_size = q.shape
    out_torch = []
    start_idx = 0
    for i in range(len(context_lengths)):
        end_idx = start_idx + context_lengths[i].item()
        torch_attn_ref_out = torch_attn_ref(
            q[start_idx:end_idx], k[start_idx:end_idx], v[start_idx:end_idx], end_idx - start_idx, num_heads, head_size
        )
        out_torch.append(torch_attn_ref_out)
        start_idx = end_idx
    return torch.cat(out_torch, dim=0)


# This method is adapted from src/transformers/models/llama/modeling_llama.py
# in transformers repository https://github.com/huggingface/transformers
# https://github.com/huggingface/transformers/blob/3b7675b2b844b02d4821b827871a21ad16dd446c/src/transformers/models/llama/modeling_llama.py#L273
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (num_tokens,
    num_key_value_heads, head_dim) to (num_tokens, num_attention_heads, head_dim)
    """
    num_tokens, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :].expand(num_tokens, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(num_tokens, num_key_value_heads * n_rep, head_dim)


@pytest.mark.skipif(not (HAS_TRITON and TRITON_CUDA_SUPPORT), reason="requires triton")
@pytest.mark.parametrize("bsz", [4, 7, 32])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("max_num_blocks_per_seq", [8, 32])
@pytest.mark.parametrize("num_attn_heads", [16])
@pytest.mark.parametrize("kv_group_num", [1, 2, 16])
@pytest.mark.parametrize("same_context_len", [True, False])
def test_context_attention(
    bsz: int,
    block_size: int,
    max_num_blocks_per_seq: int,
    num_attn_heads: int,
    kv_group_num: int,
    same_context_len: bool,
):
    torch.manual_seed(123)

    dtype = torch.float16
    device = get_current_device()
    num_seqs = bsz
    num_kv_heads = num_attn_heads // kv_group_num
    assert isinstance(num_kv_heads, int) and num_kv_heads > 0, "Invalid number of kv heads."
    head_size = 32
    max_seq_len = max_num_blocks_per_seq * block_size

    # It's necessary to clear cache here.
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    if same_context_len:
        context_lengths = torch.tensor([max_seq_len for _ in range(num_seqs)], dtype=torch.int32, device=device)
    else:
        context_lengths = torch.randint(low=1, high=max_seq_len, size=(num_seqs,), dtype=torch.int32, device=device)
    num_tokens = torch.sum(context_lengths).item()

    qkv_size = (num_tokens, num_attn_heads + 2 * num_kv_heads, head_size)
    qkv = torch.empty(size=qkv_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    q, k, v = torch.split(qkv, [num_attn_heads, num_kv_heads, num_kv_heads], dim=-2)

    cache_shape = (bsz * max_num_blocks_per_seq, num_kv_heads, head_size, block_size)
    k_cache_torch = torch.zeros(size=cache_shape, dtype=dtype, device=device)
    k_cache_triton = torch.zeros_like(k_cache_torch)
    v_cache_torch = torch.zeros(size=cache_shape, dtype=dtype, device=device)
    v_cache_triton = torch.zeros_like(v_cache_torch)

    # Mock allocation on block tables
    block_id = 0
    block_tables = torch.full(size=(num_seqs, max_num_blocks_per_seq), fill_value=-1, dtype=torch.int32)
    num_tokens_processed = 0
    for i, seq_len in enumerate(context_lengths.tolist()):
        right_bound = (seq_len + block_size - 1) // block_size  # open bound
        block_tables[i, :right_bound] = torch.arange(block_id, block_id + right_bound, dtype=torch.int32)
        # Manually fill k_cache_torch and v_cache_torch by copying from k and v
        for i in range(right_bound):
            if i == right_bound - 1:
                allocated_locs = seq_len % block_size or block_size
            else:
                allocated_locs = block_size
            k_block = k[num_tokens_processed : num_tokens_processed + allocated_locs, :, :].permute(1, 2, 0)
            v_block = v[num_tokens_processed : num_tokens_processed + allocated_locs, :, :].permute(1, 2, 0)
            cur_block_size_occupied = k_block.shape[-1]
            assert cur_block_size_occupied <= block_size, "Invalid occupied size of block during mock allocation"
            k_cache_torch[block_id, :, :, :cur_block_size_occupied] = k_block
            v_cache_torch[block_id, :, :, :cur_block_size_occupied] = v_block

            num_tokens_processed += allocated_locs
            block_id += 1

    block_tables = block_tables.to(device=device)
    out_triton = context_attention_unpadded(
        q, k, v, k_cache_triton, v_cache_triton, context_lengths, block_tables, block_size
    )

    # For GQA and MQA, repeat k, v for torch attention calculation
    # k/v won't change if provided `num_kv_group` is 1
    num_kv_group = num_attn_heads // num_kv_heads
    k = repeat_kv(k, num_kv_group)
    v = repeat_kv(v, num_kv_group)
    out_torch = torch_attn_unpad(q, k, v, context_lengths)

    assert out_torch.shape == out_triton.shape
    assert torch.allclose(out_torch, out_triton, atol=1e-2, rtol=1e-3)
    assert torch.allclose(k_cache_torch, k_cache_triton)
    assert torch.allclose(v_cache_torch, v_cache_triton)
