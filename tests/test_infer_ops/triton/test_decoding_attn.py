import pytest
import torch
from packaging import version

from colossalai.kernel.triton import flash_decoding_fwd
from colossalai.utils import get_current_device
from tests.test_infer_ops.triton.kernel_utils import mock_alloc_block_table_and_kvcache, torch_attn_ref

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


def torch_decoding(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context_lengths: torch.Tensor):
    assert context_lengths.dim() == 1, "context_lengths should be a 1D tensor"
    assert q.size(1) == 1, "Only used for decoding"
    assert k.shape == v.shape

    bsz, _, num_heads, head_dim = q.shape
    _, kv_seq_len, num_kv_heads, _ = k.shape
    assert num_heads % num_kv_heads == 0, "Invalid kv heads and attention heads."
    padding_mask = torch.zeros((bsz, 1, 1, kv_seq_len), dtype=torch.float32, device=q.device)
    for i in range(bsz):
        cur_seq_len = context_lengths[i].item()
        assert cur_seq_len <= kv_seq_len
        padding_mask[i, :, :, : kv_seq_len - cur_seq_len] = float("-inf")

    out = torch_attn_ref(q, k, v, padding_mask, bsz, 1, kv_seq_len, num_heads, num_kv_heads, head_dim)
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

    num_kv_heads = num_attn_heads // kv_group_num
    assert isinstance(num_kv_heads, int) and num_kv_heads > 0, "Invalid number of kv heads."
    q_len = 1
    head_dim = 128
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
    # Mock allocation on block tables as well as blocked kv caches
    block_tables = mock_alloc_block_table_and_kvcache(
        k, v, k_cache, v_cache, context_lengths, bsz, max_num_blocks_per_seq, block_size
    )
    block_tables = block_tables.to(device=device)

    q = q.view(bsz, q_len, num_attn_heads, head_dim)
    out_triton = flash_decoding_fwd(
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
