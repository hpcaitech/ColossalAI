import os

import pytest
import torch
from packaging import version

from colossalai.inference.kv_cache import MemoryManager
from colossalai.logging import disable_existing_loggers
from colossalai.testing import rerun_if_address_is_in_use, spawn

BATCH_SIZE = 4
INPUT_LEN = 16
OUTPUT_LEN = 8
LAYER_NUM = 4
HEAD_NUM = 32
HEAD_DIM = 128

CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.5")


def create_cache_manager(rank, world_size, port, batch_size, input_len, output_len, layer_num, head_num, head_dim):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    disable_existing_loggers()

    size = batch_size * (input_len + output_len)
    kvcache_manager = MemoryManager(size, torch.float16, head_num // world_size, head_dim, layer_num, rank)
    key_buffers = kvcache_manager.key_buffer
    value_buffers = kvcache_manager.value_buffer
    assert len(key_buffers) == len(value_buffers) == layer_num
    assert key_buffers[0].shape == value_buffers[0].shape
    # required size exceeds the maximum allocated size
    invalid_locs = kvcache_manager.alloc_contiguous(size + 1)
    assert invalid_locs is None
    # for prefill stage, allocation via alloc and alloc_contiguous should be the same
    total_token_prefill = batch_size * input_len
    prefill_locs = kvcache_manager.alloc(total_token_prefill)
    kvcache_manager.free_all()
    prefill_locs_contiguous = kvcache_manager.alloc_contiguous(total_token_prefill)[0]
    assert torch.equal(prefill_locs, prefill_locs_contiguous)
    assert torch.sum(kvcache_manager.mem_state).item() == size - total_token_prefill
    kvcache_manager.alloc_contiguous(batch_size)
    assert torch.all(kvcache_manager.mem_state[: total_token_prefill + batch_size] == False)


@pytest.mark.skipif(not CUDA_SUPPORT, reason="kv-cache manager engine requires cuda version to be higher than 11.5")
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_cache_manager_dist():
    spawn(
        create_cache_manager,
        4,
        batch_size=BATCH_SIZE,
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        layer_num=LAYER_NUM,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
    )


if __name__ == "__main__":
    test_cache_manager_dist()
