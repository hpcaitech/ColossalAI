import pytest
import torch
import torch.nn.functional as F
from packaging import version

from colossalai.kernel.triton import decoding_attention_unpadded
from colossalai.utils import get_current_device

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


def torch_attn_ref(
    q: torch.Tensor,  # [bsz, num_heads, 1, head_dim]
    k: torch.Tensor,  # [bsz, num_heads, kv_seq_len, head_dim]
    v: torch.Tensor,  # [bsz, num_heads, kv_seq_len, head_dim]
    attention_mask: torch.Tensor,
    bsz: int,
    seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    head_dim: int,
):
    qk = torch.matmul(q, k.transpose(2, 3))
    attn_scores = qk / (head_dim**0.5)

    assert attn_scores.shape == (bsz, num_heads, seq_len, kv_seq_len), "Invalid shape of attention scores"
    # for left-side padding
    if attention_mask.size() != (bsz, 1, seq_len, kv_seq_len):
        raise ValueError(
            f"Attention mask should be of size {(bsz, 1, seq_len, kv_seq_len)}, but is {attention_mask.size()}"
        )

    attn_scores = attn_scores + attention_mask
    attn_weights = F.softmax(attn_scores.to(dtype=torch.float32), dim=-1).to(dtype=q.dtype)
    out = torch.matmul(attn_weights, v)
    if out.size() != (bsz, num_heads, seq_len, head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, seq_len, head_dim)}, but is" f" {out.size()}"
        )
    out = out.transpose(1, 2).contiguous()
    return out


# This method is adapted from src/transformers/models/llama/modeling_llama.py
# in huggingface transformers repository
# https://github.com/huggingface/transformers/blob/3b7675b2b844b02d4821b827871a21ad16dd446c/src/transformers/models/llama/modeling_llama.py#L273
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (bsz, num_key_value_heads, seq_len, head_dim) to (bsz, num_attention_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    bsz, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_key_value_heads * n_rep, seq_len, head_dim)


def torch_decoding(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context_lengths: torch.Tensor):
    assert context_lengths.dim() == 1, "context_lengths should be a 1D tensor"
    assert q.size(1) == 1, "Only used for decoding"
    assert k.shape == v.shape

    bsz, _, num_heads, head_dim = q.shape
    _, kv_seq_len, num_kv_heads, _ = k.shape
    assert num_heads % num_kv_heads == 0, "Invalid kv heads and attention heads."
    kv_group_num = num_heads // num_kv_heads
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    k = repeat_kv(k, kv_group_num)
    v = repeat_kv(v, kv_group_num)
    padding_mask = torch.zeros((bsz, 1, 1, kv_seq_len), dtype=torch.float32, device=q.device)
    for i in range(bsz):
        cur_seq_len = context_lengths[i].item()
        assert cur_seq_len <= kv_seq_len
        padding_mask[i, :, :, : kv_seq_len - cur_seq_len] = float("-inf")

    out = torch_attn_ref(q, k, v, padding_mask, bsz, 1, kv_seq_len, num_heads, head_dim)
    return out


@pytest.mark.skipif(not (HAS_TRITON and TRITON_CUDA_SUPPORT), reason="requires triton")
@pytest.mark.parametrize("bsz", [4, 7, 32])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("max_num_blocks_per_seq", [8, 32])
@pytest.mark.parametrize("num_attn_heads", [16])
@pytest.mark.parametrize("kv_group_num", [1, 2, 16])
@pytest.mark.parametrize("same_context_len", [True, False])
def test_flash_decoding(
    bsz: int,
    block_size: int,
    max_num_blocks_per_seq: int,
    num_attn_heads: int,
    kv_group_num: int,
    same_context_len: bool,
):
    torch.manual_seed(123)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    head_dim = 128
    q_len = 1
    num_kv_heads = num_attn_heads // kv_group_num
    assert isinstance(num_kv_heads, int) and num_kv_heads > 0, "Invalid number of kv heads."
    max_seq_len = block_size * max_num_blocks_per_seq
    dtype = torch.float16
    device = get_current_device()

    if same_context_len:
        context_lengths = torch.tensor([max_seq_len for _ in range(bsz)], dtype=torch.int32, device=device)
    else:
        context_lengths = torch.randint(low=1, high=max_seq_len, size=(bsz,), dtype=torch.int32, device=device)
    num_tokens = torch.sum(context_lengths).item()

    q_size = (bsz, q_len, num_attn_heads, head_dim)
    q = torch.empty(size=q_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    kv_size = (num_tokens, 2 * num_kv_heads, head_dim)
    kv = torch.empty(size=kv_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    k, v = torch.split(kv, [num_kv_heads, num_kv_heads], dim=-2)

    cache_shape = (bsz * max_num_blocks_per_seq, num_kv_heads, head_dim, block_size)
    k_cache = torch.zeros(size=cache_shape, dtype=dtype, device=device)
    v_cache = torch.zeros(size=cache_shape, dtype=dtype, device=device)
    # Mock allocation on block tables and blocked kv caches
    block_id = 0
    block_tables = torch.full(size=(bsz, max_num_blocks_per_seq), fill_value=-1, dtype=torch.int32)
    num_tokens_processed = 0
    for i, seq_len in enumerate(context_lengths.tolist()):
        right_bound = (seq_len + block_size - 1) // block_size  # open bound
        block_tables[i, :right_bound] = torch.arange(block_id, block_id + right_bound, dtype=torch.int32)
        # Manually fill kv caches by copying from k and v
        for i in range(right_bound):
            if i == right_bound - 1:
                allocated_locs = seq_len % block_size or block_size
            else:
                allocated_locs = block_size
            k_block = k[num_tokens_processed : num_tokens_processed + allocated_locs, :, :].permute(1, 2, 0)
            v_block = v[num_tokens_processed : num_tokens_processed + allocated_locs, :, :].permute(1, 2, 0)
            cur_block_size_occupied = k_block.shape[-1]
            assert cur_block_size_occupied <= block_size, "Invalid occupied size of block during mock allocation"
            k_cache[block_id, :, :, :cur_block_size_occupied] = k_block
            v_cache[block_id, :, :, :cur_block_size_occupied] = v_block

            num_tokens_processed += allocated_locs
            block_id += 1

    block_tables = block_tables.to(device=device)

    q = q.view(bsz, q_len, num_attn_heads, head_dim)
    out_triton = decoding_attention_unpadded(
        q,
        k_cache,
        v_cache,
        context_lengths,
        block_tables,
        block_size,
        kv_group_num,
    )
    out_triton = out_triton.unsqueeze(1)  # [bsz, 1, num_heads, head_dim]

    # rebuild (batched) kv with padding for torch attention
    # q   [bsz, 1, num_heads, head_dim]
    # k/v [num_tokens, num_kv_heads, head_dim]
    max_seq_len = context_lengths.max().item()
    k_torch = torch.zeros((bsz, max_seq_len, num_kv_heads, head_dim), dtype=k.dtype, device=k.device)
    v_torch = torch.zeros_like(k_torch)
    prev_len_sum = 0
    for i, seq_len in enumerate(context_lengths.tolist()):
        # mock left-side padding
        k_torch[i, -seq_len:, :, :] = k[prev_len_sum : prev_len_sum + seq_len]
        v_torch[i, -seq_len:, :, :] = v[prev_len_sum : prev_len_sum + seq_len]
        prev_len_sum += seq_len
    # k/v [bsz, max_seq_len, num_kv_heads, head_dim]
    out_torch = torch_decoding(q, k_torch, v_torch, context_lengths)

    assert out_torch.shape == out_triton.shape
    assert torch.allclose(out_torch, out_triton, atol=1e-3, rtol=1e-4)
