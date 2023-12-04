from typing import List, Tuple

import torch

from colossalai.inference.config import InferenceConfig
from colossalai.logging import get_dist_logger
from colossalai.utils import get_current_device

from .block_cache import CacheBlock

GIGABYTE = 1024**3


class KVCacheManager:
    """KVCacheManager manages both the logical cache blocks and physical KV cache (tensors).

    NOTE: The KVCacheManager is designed to be interacted with by using indices of logical blocks.
        That is, it won't allocate and return a physical cache to the engine or scheduler;
        instead, it will mark the logical block as allocated, and return, or update, the block id representing
        the physical cache to the caller. In kernels are where the physical cache is actually used and updated.

    Args:
        config(InferenceConfig): The All-in-one inference configuration.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.logger = get_dist_logger(__name__)
        self.device = get_current_device()
        # Model settings
        self.dtype = config.dtype
        self.elem_size_in_bytes = torch.tensor([], dtype=self.dtype).element_size()
        self.num_layers = config.num_layers
        self.num_heads = config.num_attention_heads
        self.head_size = config.head_size
        # Generation settings
        self.max_batch_size = config.max_batch_size
        self.max_input_length = config.max_input_length
        self.max_output_length = config.max_output_length

        # Logical cache blocks allocation
        self.block_size = config.block_size
        # NOTE: `num_blocks` is not prompted, but evaluated from the maximum input/output length, and the maximum batch size
        self.max_blocks_per_sequence = (
            self.max_input_length + self.max_output_length + self.block_size - 1
        ) // self.block_size
        self.num_blocks = self.max_blocks_per_sequence * self.max_batch_size
        self.available_blocks = self.num_blocks
        self._free_blocks = self._init_logical_caches()
        self._cache_blocks = tuple(self._free_blocks)
        self._allocated_blocks = []
        # block availablity state 0->allocated, 1->free
        self._block_states = torch.ones((self.num_blocks,), dtype=torch.bool, device="cpu")

        # Physical cache allocation
        alloc_shape = (self.num_blocks, self.block_size, self.num_heads, self.head_size)
        self.logger.info(f"Allocating KV cache with shape: {alloc_shape} consisting of {self.num_blocks} blocks.")
        # self._kv_caches = self._init_device_caches(alloc_shape)
        self._kv_caches = self._init_device_caches()
        self.total_physical_cache_size_in_bytes = (
            self.elem_size_in_bytes
            * self.num_layers
            * 2
            * self.num_blocks
            * self.block_size
            * self.num_heads
            * self.head_size
        )

    def get_total_num_blocks(self) -> int:
        """Get the total number of logical cache blocks."""
        return self.num_blocks

    def get_max_blocks_per_sequence(self) -> int:
        """Get the maximum number of blocks that can be allocated for a single sequence."""
        return self.max_blocks_per_sequence

    def allocate_from_last_block_idx(self, last_block_id: int, space_asked: int = 1) -> List[int]:
        """Allocate the logical cache blocks for a single sequence.
        It returns the allocated block ids as a list.

        Args:
            last_block_id: The last-allocated block id in the block table of the sequence.
            space_asked: i.e. The number of tokens required to assign space for.
        Returns:
            A list of allocated block ids. If the prompted last-allocated block has enough space,
            it will be the only one in the list.
        """
        blocks = []
        last_block: CacheBlock = self._cache_blocks[last_block_id]
        if last_block.has_space():
            if last_block.has_ref():
                # TODO: Should turn to use a new block
                raise NotImplementedError("Copy-On-Write is not supported yet.")
            space_asked = self._allocate_on_block(last_block, space_asked)
            blocks.append(last_block_id)
        while space_asked > 0:
            new_block: CacheBlock = self._free_blocks.pop(0)
            space_asked = self._allocate_on_block(new_block, space_asked)
            self._allocated_blocks.append(new_block)
            self.available_blocks -= 1
            blocks.append(new_block.block_id)
            self._block_states[new_block.block_id] = 0

        return blocks

    def allocate_from_block_table(self, block_table: torch.Tensor, seq_length: int, space_asked: int = 1) -> None:
        """Allocate the logical cache blocks for a single sequence.
        It updates the provided block table with the allocated block(s).

        Args:
            block_table: A 1D tensor of shape [max_blocks_per_sequence], storing mapping of token_position_id -> block_id.
            seq_length: The length of the sequence.
            space_asked: i.e. The number of tokens to be assigned space for.
        """
        assert block_table.dim() == 1
        # the last-allocated block can be fully occupied, and can be occupied only one slot as well
        last_allocated_block_local_idx = seq_length // self.block_size
        # the right-most block that to be allocated, fully or partially
        last_newly_allocated_block_local_idx = (seq_length + space_asked - 1) // self.block_size
        last_newly_allocated_block_local_idx = min(last_newly_allocated_block_local_idx, block_table.numel())

        for i in range(last_allocated_block_local_idx, last_newly_allocated_block_local_idx + 1):
            block_global_id = block_table[i].item()
            if block_global_id < 0:
                assert self.available_blocks > 0, "No available blocks to allocate."
                free_block: CacheBlock = self._free_blocks.pop(0)
                self._allocated_blocks.append(free_block)
                block_global_id = free_block.block_id
                self.available_blocks -= 1
                self._block_states[block_global_id] = 0
                block_table[i] = block_global_id

            block: CacheBlock = self._cache_blocks[block_global_id]
            space_asked = self._allocate_on_block(self, block, space_asked)

    def free_cache_blocks(self, block_table: torch.Tensor):
        """Free the logical cache blocks for **a single sequence**."""
        assert block_table.dim() == 1
        assert torch.all(block_table >= 0)
        for i in range(block_table.numel()):
            global_block_id = block_table[i].item()
            block: CacheBlock = self._cache_blocks[global_block_id]
            block.remove_ref()  # not going to clear the block thoroughly

    def get_physical_cache(self, layer_id: int, block_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the corresponding tensor for the provided block id for a specific layer."""
        return self._kv_caches[0][layer_id][block_idx], self._kv_caches[1][layer_id][block_idx]

    def _allocate_on_block(self, block: CacheBlock, space_asked: int) -> None:
        """Allocate a specific size of space on a cache block."""
        available_space = block.available_space()
        assert available_space > 0, "No available blocks to allocate."
        space_to_allocate = min(available_space, space_asked)
        block.allocate(space_to_allocate)
        return space_asked - space_to_allocate

    def _init_logical_caches(self):
        """Initialize the logical cache blocks."""
        blocks = []
        for i in range(self.num_blocks):
            cache_block = CacheBlock(i, self.block_size, self.elem_size_in_bytes)
            blocks.append(cache_block)
        return blocks

    def _init_device_caches(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize the physical cache on the device.
        For each layer of the model, we allocate two tensors for key and value respectively,
        with shape of [num_blocks, block_size, num_head, head_size]
        TODO: Explore the performance when using difference shapes with kernel-related optimizations
        """
        alloc_shape = (self.num_blocks, self.block_size, self.num_heads, self.head_size)
        k_cache: List[torch.Tensor] = []
        v_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            k_cache.append(torch.zeros(alloc_shape, dtype=self.dtype, device=self.device))
            v_cache.append(torch.zeros(alloc_shape, dtype=self.dtype, device=self.device))
        return k_cache, v_cache
