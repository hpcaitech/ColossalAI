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
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor,
    bsz: int,
    seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    head_size: int,
):
    qk = torch.matmul(q, k.transpose(2, 3))
    attn_scores = qk / (head_size**0.5)

    assert attn_scores.shape == (bsz, num_heads, seq_len, kv_seq_len), "Invalid shape of attention scores"

    # for left-side padding
    if attention_mask.size() != (bsz, 1, seq_len, kv_seq_len):
        raise ValueError(
            f"Attention mask should be of size {(bsz, 1, seq_len, kv_seq_len)}, but is {attention_mask.size()}"
        )

    attn_scores = attn_scores + attention_mask

    attn_weights = F.softmax(attn_scores.to(dtype=torch.float32), dim=-1).to(dtype=q.dtype)

    out = torch.matmul(attn_weights, v)
    if out.size() != (bsz, num_heads, seq_len, head_size):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, seq_len, head_size)}, but is" f" {out.size()}"
        )
    out = out.transpose(1, 2).contiguous()
    return out


def torch_decoding_unpad(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context_lengths: torch.Tensor):
    # E.g.
    # q torch.Size([4, 1, 16, 128])
    # k/v torch.Size([4, 64, 16, 128])
    assert context_lengths.dim() == 1, "context_lengths should be a 1D tensor"
    assert q.size(1) == 1, "only used for decoding"
    assert k.shape == v.shape

    bsz, _, num_heads, head_size = q.shape
    _, kv_seq_len, num_kv_heads, _ = k.shape
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    padding_mask = torch.zeros((bsz, 1, 1, kv_seq_len), dtype=torch.float32, device=q.device)
    for i in range(bsz):
        cur_seq_len = context_lengths[i].item()
        assert cur_seq_len <= kv_seq_len
        padding_mask[i, :, :, : kv_seq_len - cur_seq_len] = float("-inf")

    out = torch_attn_ref(q, k, v, padding_mask, bsz, 1, kv_seq_len, num_heads, head_size)
    return out


if __name__ == "__main__":
    torch.manual_seed(123)

    # to be moved to a func
    bsz = 4
    block_size = 16
    max_num_blocks_per_seq = 4
    num_attn_heads = num_kv_heads = 16
    head_size = 128
    same_context_len = False
    max_seq_len = block_size * max_num_blocks_per_seq

    q_len = 1
    num_seqs = bsz
    kv_group_num = num_attn_heads // num_kv_heads
    dtype = torch.float16
    device = get_current_device()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    if same_context_len:
        context_lengths = torch.tensor([max_seq_len for _ in range(num_seqs)], dtype=torch.int32, device=device)
    else:
        context_lengths = torch.randint(low=1, high=max_seq_len, size=(num_seqs,), dtype=torch.int32, device=device)
    num_tokens = torch.sum(context_lengths).item()

    q_size = (bsz, q_len, num_attn_heads, head_size)
    q = torch.empty(size=q_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    kv_size = (num_tokens, 2 * num_kv_heads, head_size)
    kv = torch.empty(size=kv_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    k, v = torch.split(kv, [num_kv_heads, num_kv_heads], dim=-2)

    cache_shape = (bsz * max_num_blocks_per_seq, num_kv_heads, head_size, block_size)
    k_cache = torch.zeros(size=cache_shape, dtype=dtype, device=device)
    v_cache = torch.zeros(size=cache_shape, dtype=dtype, device=device)

    # Mock allocation on block tables
    block_id = 0
    block_tables = torch.full(size=(num_seqs, max_num_blocks_per_seq), fill_value=-1, dtype=torch.int32)
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

    q = q.view(bsz, q_len, num_attn_heads, head_size)
    out_triton = decoding_attention_unpadded(
        q,
        k_cache,
        v_cache,
        context_lengths,
        block_tables,
        block_size,
        num_kv_group=1,
    )
    out_triton = out_triton.unsqueeze(1)

    # q [bsz, 1, num_heads, head_size]
    # k/v [num_tokens, num_kv_heads, head_size]
    # rebuild kv
    max_seq_len = context_lengths.max().item()
    k_torch = torch.zeros((bsz, max_seq_len, num_kv_heads, head_size), dtype=k.dtype, device=k.device)
    v_torch = torch.zeros_like(k_torch)
    prev_len_sum = 0
    for i, seq_len in enumerate(context_lengths.tolist()):
        # mock left-side padding
        k_torch[i, -seq_len:, :, :] = k[prev_len_sum : prev_len_sum + seq_len]
        v_torch[i, -seq_len:, :, :] = v[prev_len_sum : prev_len_sum + seq_len]
        prev_len_sum += seq_len
    # k/v [bsz, max_seq_len, num_kv_heads, head_size]
    out_torch = torch_decoding_unpad(q, k_torch, v_torch, context_lengths)

    assert out_torch.shape == out_triton.shape
    assert torch.allclose(out_torch, out_triton, atol=1e-3, rtol=1e-4)
