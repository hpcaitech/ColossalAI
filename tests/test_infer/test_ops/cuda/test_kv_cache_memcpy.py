import pytest
import torch

from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.utils import get_current_device
from tests.test_infer.test_ops.triton.test_kvcache_copy import prepare_data

inference_ops = InferenceOpsLoader().load()

HEAD_DIM = 4


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


if __name__ == "__main__":
    test_copy_kv_to_caches(4, 32, 8, 16, True)
