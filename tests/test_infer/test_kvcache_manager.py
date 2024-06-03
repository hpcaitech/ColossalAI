import random

import pytest
import torch
from transformers.models.llama import LlamaConfig

import colossalai
from colossalai.inference.config import InferenceConfig
from colossalai.inference.kv_cache import CacheBlock, KVCacheManager
from colossalai.logging import disable_existing_loggers
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


@parameterize(
    "test_config",
    [
        {
            "elem_size": 2,
            "block_size": 4,
        }
    ],
)
def test_logical_blocks(test_config):
    block = CacheBlock(block_id=0, block_size=test_config["block_size"], elem_size=test_config["elem_size"])

    assert block.is_empty()
    assert block.available_space == test_config["block_size"]
    assert not block.has_ref()
    block.add_ref()
    assert block.ref_count == 1
    assert block.has_ref()
    block.remove_ref()
    assert block.ref_count == 0
    block.allocate(1)
    assert block.allocated_size == 1
    block.allocate(test_config["block_size"] - 1)
    assert block.available_space < 1


@parameterize(
    "test_config",
    [
        {
            "hidden_size": 512,
            "num_attention_heads": 16,
            "num_layers": 2,
            "block_size": 8,
            "max_batch_size": 10,
            "max_input_len": 32,
            "max_output_len": 32,
            "dtype": torch.float32,
            "beam_width": 1,
            "tp_size": 1,
        },
        {
            "hidden_size": 128,
            "num_attention_heads": 4,
            "num_layers": 3,
            "block_size": 4,
            "max_batch_size": 4,
            "max_input_len": 64,
            "max_output_len": 32,
            "dtype": torch.float16,
            "beam_width": 3,
            "tp_size": 1,
        },
    ],
)
def check_cache_manager(test_config):
    disable_existing_loggers()

    assert test_config["max_batch_size"] > 1

    hidden_size = test_config.pop("hidden_size")
    num_layers = test_config.pop("num_layers")
    num_attention_heads = test_config.pop("num_attention_heads")
    head_size = hidden_size // num_attention_heads
    block_size = test_config["block_size"]
    max_batch_size = test_config["max_batch_size"]
    max_input_length = test_config["max_input_len"]
    max_output_length = test_config["max_output_len"]

    inference_config = InferenceConfig(**test_config)
    model_config = LlamaConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
    )
    cache_manager = KVCacheManager(inference_config, model_config)

    num_blocks = cache_manager.total_num_blocks
    assert num_blocks > 0
    assert len(cache_manager._cache_blocks) == num_blocks
    key_caches = cache_manager._kv_caches[0]  # key caches for all the blocks in all the layers
    assert len(key_caches) == num_layers
    expected_kv_shape = (num_blocks, num_attention_heads, block_size, head_size)
    assert key_caches[0].shape == expected_kv_shape
    k_cache_block0, v_cache_block0 = cache_manager.get_physical_cache(0, 0)
    expected_kv_block_shape = expected_kv_shape[1:]
    assert k_cache_block0.shape == expected_kv_block_shape
    assert v_cache_block0.shape == expected_kv_block_shape

    max_blocks_per_seq = cache_manager.get_max_blocks_per_sequence()
    block_tables = torch.tensor(
        [[-1 for _ in range(max_blocks_per_seq)] for _ in range(test_config["max_batch_size"])], dtype=torch.int32
    )
    context_lengths = [random.randint(1, max_input_length) for _ in range(max_batch_size)]
    cnt_blocks_used = 0
    # Mock Prefill
    for req_i in range(max_batch_size):
        cur_seq_len = context_lengths[req_i]
        cur_block_table = block_tables[req_i]
        cache_manager.allocate_context_from_block_table(cur_block_table, cur_seq_len)
        last_allocated_idx = (cur_seq_len - 1) // block_size
        assert torch.all(cur_block_table[: last_allocated_idx + 1] >= 0)
        cnt_blocks_used += torch.sum(cur_block_table >= 0).item()
    assert cache_manager.num_available_blocks == num_blocks - cnt_blocks_used

    # Mock Decoding
    for req_i in range(max_batch_size):
        context_length = context_lengths[req_i]
        cur_output_length = random.randint(1, max_output_length)
        cur_block_table = block_tables[req_i]
        for _ in range(cur_output_length):
            cache_manager.allocate_token_from_block_table(cur_block_table, context_length)
            context_length += 1
        context_length -= 1
        last_allocated_idx = context_length // block_size
        space_allocated_on_last_block = context_length % block_size + 1
        assert space_allocated_on_last_block > 0
        block_id = cur_block_table[last_allocated_idx]
        block: CacheBlock = cache_manager._cache_blocks[block_id]
        assert block.allocated_size == space_allocated_on_last_block

    # Randomly select a request and clear its cache
    req_i = random.randint(0, max_batch_size - 1)
    context_length = context_lengths[req_i]
    blocks_used_by_req = torch.sum(block_tables[req_i] >= 0).item()
    prev_available_blocks = cache_manager.num_available_blocks
    cache_manager.free_block_table(block_tables[req_i])
    assert cache_manager.num_available_blocks == blocks_used_by_req + prev_available_blocks

    k_ptr_block0_layer0, _ = cache_manager.get_block_kv_ptrs(0, 0)
    k_ptr_block1_layer0, _ = cache_manager.get_block_kv_ptrs(1, 0)
    elem_size = torch.tensor([], dtype=test_config["dtype"]).element_size()
    expected_stride = block_size * num_attention_heads * head_size * elem_size
    assert k_ptr_block1_layer0 - k_ptr_block0_layer0 == expected_stride
    cache_manager.clear_all()
    assert cache_manager.num_available_blocks == num_blocks

    for cache_block in cache_manager._cache_blocks:
        assert cache_block.available_space == block_size

    # Mock batch operations (Prefill/Decoding updates)
    context_lengths = torch.tensor([max_input_length, max_input_length - 1])
    block_tables = torch.tensor(
        [[-1 for _ in range(cache_manager.max_blocks_per_sequence)] for _ in range(2)], dtype=torch.int32
    )
    cache_manager.allocate_context_from_block_tables(block_tables, context_lengths)
    cache_manager.allocate_tokens_from_block_tables(block_tables, context_lengths)
    cache_manager.free_block_tables(block_tables)
    for cache_block in cache_manager._cache_blocks:
        assert cache_block.available_space == block_size


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_cache_manager()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_cache_manager():
    spawn(run_dist, 1)


if __name__ == "__main__":
    test_logical_blocks()
    test_cache_manager()
