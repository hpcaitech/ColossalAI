import pytest
import torch
from packaging import version

from colossalai.kernel.triton import copy_kv_to_blocked_cache
from colossalai.utils import get_current_device
from tests.test_infer_ops.triton.kernel_utils import mock_alloc_block_table_and_kvcache, mock_alloc_single_token

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


@pytest.mark.skipif(not (HAS_TRITON and TRITON_CUDA_SUPPORT), reason="requires triton")
@pytest.mark.parametrize("bsz", [4, 7, 32])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("max_num_blocks_per_seq", [8, 32])
@pytest.mark.parametrize("num_kv_heads", [16])
@pytest.mark.parametrize("same_context_len", [True, False])
def test_copy_kv_to_caches(
    bsz: int,
    block_size: int,
    max_num_blocks_per_seq: int,
    num_kv_heads: int,
    same_context_len: bool,
):
    torch.manual_seed(123)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    head_dim = 128
    max_seq_len = block_size * max_num_blocks_per_seq
    dtype = torch.float16
    device = get_current_device()

    if same_context_len:
        # context_lengths in this test records the previous kv seq len
        # (not incorporating the current input whose seq len is 1)
        context_lengths = torch.tensor([max_seq_len - 1 for _ in range(bsz)], dtype=torch.int32, device=device)
    else:
        context_lengths = torch.randint(low=1, high=max_seq_len - 1, size=(bsz,), dtype=torch.int32, device=device)
    num_tokens = torch.sum(context_lengths).item()

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

    new_k = torch.randn((bsz, 1, num_kv_heads, head_dim), dtype=dtype, device=device)
    # mock allocating blocks for the new k/v and update block tables
    mock_alloc_single_token(block_tables, context_lengths, block_size)
    copy_kv_to_blocked_cache(new_k, k_cache, context_lengths, block_tables)

    for seq_i in range(bsz):
        ki = new_k[seq_i]
        ki = ki.squeeze()
        context_len_i = context_lengths[seq_i]
        target_block_id = block_tables[seq_i, context_len_i // block_size]
        offsets_in_block = context_len_i % block_size
        target = k_cache[target_block_id, :, :, offsets_in_block]
        orig = new_k[seq_i].squeeze(dim=0)
        assert torch.equal(orig, target)


if __name__ == "__main__":
    test_copy_kv_to_caches(4, 32, 8, 16, False)
