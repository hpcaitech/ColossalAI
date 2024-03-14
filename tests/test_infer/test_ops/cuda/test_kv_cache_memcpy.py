import pytest
import torch
import torch.nn.functional as F

from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.utils import get_current_device
from tests.test_infer.test_ops.triton.kernel_utils import generate_caches_and_block_tables_v2
from tests.test_infer.test_ops.triton.test_kvcache_copy import prepare_data

inference_ops = InferenceOpsLoader().load()

HEAD_DIM = 4


def run_decode_copy_kv_to_caches(
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

    max_seq_len = block_size * max_num_blocks_per_seq
    dtype = torch.float32
    device = get_current_device()

    new_k, new_v, k_cache, v_cache, kv_seq_lengths, block_tables = prepare_data(
        bsz,
        num_kv_heads,
        HEAD_DIM,
        block_size,
        max_num_blocks_per_seq,
        same_context_len,
        max_seq_len,
        device=device,
        dtype=dtype,
    )

    new_k = new_k.squeeze(1) if new_k.dim() == 4 else new_k
    new_v = new_v.squeeze(1) if new_v.dim() == 4 else new_v
    inference_ops.decode_kv_cache_memcpy(new_k, new_v, k_cache, v_cache, kv_seq_lengths, block_tables)

    past_kv_seq_len = kv_seq_lengths - 1
    target_block_ids = block_tables[range(0, block_tables.size(0)), past_kv_seq_len // block_size]
    offsets_in_block = past_kv_seq_len % block_size
    k_target = k_cache[target_block_ids, :, offsets_in_block, :]
    k_source = new_k.squeeze()
    v_target = v_cache[target_block_ids, :, offsets_in_block, :]
    v_source = new_v.squeeze()

    assert k_target.shape == k_source.shape
    assert torch.equal(k_target, k_source)
    assert v_target.shape == v_source.shape
    assert torch.equal(v_target, v_source)


def run_context_copy_kv_to_cache(
    bsz: int,
    block_size: int,
    max_num_blocks_per_seq: int,
    num_kv_heads: int,
    same_context_len: bool,
):
    torch.manual_seed(123)

    assert isinstance(num_kv_heads, int) and num_kv_heads > 0, "Invalid number of kv heads."
    max_seq_len = max_num_blocks_per_seq * block_size
    dtype = torch.float16
    device = get_current_device()

    if same_context_len:
        context_lengths = torch.tensor([max_seq_len for _ in range(bsz)], dtype=torch.int32, device=device)
    else:
        context_lengths = torch.randint(low=1, high=max_seq_len, size=(bsz,), dtype=torch.int32, device=device)

    num_tokens = torch.sum(context_lengths).item()

    max_seq_len_in_batch = context_lengths.max()
    cu_seqlens = F.pad(torch.cumsum(context_lengths, dim=0, dtype=torch.torch.int32), (1, 0))

    kv_size = (num_tokens, num_kv_heads, HEAD_DIM)
    key = torch.empty(size=kv_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    value = torch.empty(size=kv_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)

    k_cache_ref, v_cache_ref, block_tables = generate_caches_and_block_tables_v2(
        key, value, context_lengths, bsz, max_num_blocks_per_seq, block_size, dtype, device
    )

    block_tables = block_tables.to(device=device)
    k_cache = torch.zeros_like(k_cache_ref)
    v_cache = torch.zeros_like(v_cache_ref)

    inference_ops.context_kv_cache_memcpy(
        key, value, k_cache, v_cache, context_lengths, cu_seqlens, block_tables, max_seq_len_in_batch
    )

    assert torch.equal(k_cache, k_cache_ref)
    assert torch.equal(v_cache, v_cache_ref)


@pytest.mark.parametrize("bsz", [4, 7, 32])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("max_num_blocks_per_seq", [8, 32])
@pytest.mark.parametrize("num_kv_heads", [16])
@pytest.mark.parametrize("same_context_len", [True, False])
def test_kv_cache_memcopy(
    bsz: int,
    block_size: int,
    max_num_blocks_per_seq: int,
    num_kv_heads: int,
    same_context_len: bool,
):
    run_context_copy_kv_to_cache(bsz, block_size, max_num_blocks_per_seq, num_kv_heads, same_context_len)
    run_decode_copy_kv_to_caches(bsz, block_size, max_num_blocks_per_seq, num_kv_heads, same_context_len)


if __name__ == "__main__":
    test_kv_cache_memcopy(4, 32, 8, 16, True)
