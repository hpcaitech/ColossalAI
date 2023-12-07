from dataclasses import dataclass

import torch
from transformers.models.llama import LlamaConfig

from colossalai.inference.kv_cache import CacheBlock, KVCacheManager
from colossalai.logging import disable_existing_loggers
from colossalai.testing import parameterize


@dataclass
class SampleInferenceConfig:
    # This struct is only used for testing KVCache Manager.
    # The inference config must have the following attributes to initialize a KVCacheManager.
    block_size: int
    max_batch_size: int
    max_input_len: int
    max_output_len: int
    beam_width: int
    dtype: torch.dtype
    tp_size: int


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
            "max_batch_size": 9,
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
def test_cache_manager(test_config):
    disable_existing_loggers()

    hidden_size = test_config.pop("hidden_size")
    num_layers = test_config.pop("num_layers")
    num_attention_heads = test_config.pop("num_attention_heads")
    head_size = hidden_size // num_attention_heads
    block_size = test_config["block_size"]
    max_input_length = test_config["max_input_len"]
    max_output_length = test_config["max_output_len"]

    sample_config = SampleInferenceConfig(**test_config)
    model_config = LlamaConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
    )
    cache_manager = KVCacheManager(sample_config, model_config)

    num_blocks = cache_manager.get_total_num_blocks()
    assert num_blocks > 0
    assert len(cache_manager._cache_blocks) == len(cache_manager._free_blocks) == num_blocks
    assert len(cache_manager._allocated_blocks) == 0
    key_caches = cache_manager._kv_caches[0]  # key caches for all the blocks in all the layers
    assert len(key_caches) == num_layers
    expected_kv_shape = (num_blocks, num_attention_heads, head_size, block_size)
    assert key_caches[0].shape == expected_kv_shape
    k_cache_block0, v_cache_block0 = cache_manager.get_physical_cache(0, 0)
    expected_kv_block_shape = expected_kv_shape[1:]
    assert k_cache_block0.shape == expected_kv_block_shape
    assert v_cache_block0.shape == expected_kv_block_shape

    max_blocks_per_seq = cache_manager.get_max_blocks_per_sequence()
    block_table = torch.tensor([-1 for _ in range(max_blocks_per_seq)], dtype=torch.int32)
    sequence_length = max_input_length
    cache_manager.allocate_from_block_table(block_table, 0, sequence_length)
    last_allocated_idx = (sequence_length - 1) // block_size
    assert torch.all(block_table[: last_allocated_idx + 1] >= 0)

    max_length = max_input_length + max_output_length
    while sequence_length < max_length:
        cache_manager.allocate_from_block_table(block_table, sequence_length, 1)
        sequence_length += 1
        last_allocated_idx = (sequence_length - 1) // block_size
        num_fully_allocated_blocks = sequence_length // block_size
        space_allocated_on_last_block = sequence_length % block_size
        space_allocated_on_last_block = (
            block_size if last_allocated_idx != num_fully_allocated_blocks else space_allocated_on_last_block
        )
        assert space_allocated_on_last_block > 0
        block_id = block_table[last_allocated_idx]
        block: CacheBlock = cache_manager._cache_blocks[block_id]
        assert block.allocated_size == space_allocated_on_last_block

    assert torch.all(block_table >= 0)
    assert cache_manager.available_blocks < num_blocks

    k_ptr_block0_layer0, _ = cache_manager.get_block_kv_ptrs(0, 0)
    k_ptr_block1_layer0, _ = cache_manager.get_block_kv_ptrs(1, 0)
    elem_size = torch.tensor([], dtype=test_config["dtype"]).element_size()
    expected_stride = block_size * num_attention_heads * head_size * elem_size
    assert k_ptr_block1_layer0 - k_ptr_block0_layer0 == expected_stride

    cache_manager.free_cache_blocks(block_table)
    assert torch.all(block_table < 0)
    assert cache_manager.available_blocks == num_blocks


if __name__ == "__main__":
    test_logical_blocks()
    test_cache_manager()
