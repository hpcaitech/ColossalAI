from typing import List, Tuple

import torch
from transformers.configuration_utils import PretrainedConfig

from colossalai.inference.config import InferenceConfig
from colossalai.logging import get_dist_logger
from colossalai.utils import get_current_device

from .block_cache import CacheBlock

GIGABYTE = 1024**3


def get_model_config_attr(config: PretrainedConfig, attr_name: str):
    if hasattr(config, attr_name):
        return getattr(config, attr_name)
    elif hasattr(config, "attribute_map") and hasattr(config, config.attribute_map[attr_name]):
        return getattr(config, config.attribute_map[attr_name])
    raise AttributeError(f"{attr_name} is not found in config")


class KVCacheManager:
    """KVCacheManager manages both the logical cache blocks and physical KV cache (tensors).

    NOTE: The KVCacheManager is designed to be interacted with indices of logical blocks.
        That is, it won't allocate and return a physical cache to the engine or scheduler;
        instead, it will mark the logical block as allocated and update the block id representing
        the physical cache to the caller. The physical cache is actually used and updated in kernels.

    Example
        A block table of a single sequence before block allocation might be:
        | -1 | -1 | -1 | -1 | -1 | -1 |
        where the maximum blocks per sequence is 6
        The block table after block allocation might be:
        |  0 |  1 |  2 | -1 | -1 | -1 |
        Then the logical blocks with id 0, 1, and 2, are allocated for this sequence,
        and the physical caches, each with size of `block_size * head_num * head_size * elem_size` for a single layer,
        corresponding to these blocks will be used to read/write KV Caches in kernels.

        For a batch of sequences, the block tables after allocation might be:
        |  0 |  1 |  2 | -1 | -1 | -1 |
        |  3 |  4 |  5 |  6 |  7 | -1 |
        |  8 |  9 | 10 | 11 | -1 | -1 |
        | 12 | 13 | 14 | 15 | -1 | -1 |
        where 16 logical cache blocks are allocated and the same number of physical cache blocks will be used in kernels.

        Currently, allocations and updates are done at granularity of a single sequence.
        That is, the block table should be a 1D tensor of shape [max_blocks_per_sequence].
        And it's possible to have a batch of sequences with different lengths of block tables.
    """

    def __init__(self, config: InferenceConfig, model_config: PretrainedConfig, verbose: bool = False) -> None:
        self.logger = get_dist_logger(__name__)
        self.device = get_current_device()

        # Parallel settings
        self.tp_size = config.tp_size
        # Model settings
        self.dtype = config.dtype
        self.elem_size_in_bytes = torch.tensor([], dtype=self.dtype).element_size()
        self.num_layers = get_model_config_attr(model_config, "num_hidden_layers")
        # For now we focus on MHA only, TODO add handling for MQA and GQA
        self.head_num = get_model_config_attr(model_config, "num_attention_heads")
        self.head_size = get_model_config_attr(model_config, "hidden_size") // self.head_num
        assert self.head_num % self.tp_size == 0, f"Cannot shard {self.head_num} heads with tp size {self.tp_size}"
        self.head_num //= self.tp_size
        self.beam_width = config.beam_width
        self.max_batch_size = config.max_batch_size
        self.max_input_length = config.max_input_len
        self.max_output_length = config.max_output_len
        # Cache block settings
        self.block_size = config.block_size
        # NOTE: `num_blocks` is not prompted, but evaluated from the maximum input/output length, and the maximum batch size
        self.max_blocks_per_sequence = (
            self.max_input_length + self.max_output_length + self.block_size - 1
        ) // self.block_size
        self.num_blocks = self.max_blocks_per_sequence * self.max_batch_size * self.beam_width

        # Physical cache allocation
        if verbose:
            alloc_shape = (self.num_blocks, self.head_num, self.head_size, self.block_size)
            self.logger.info(f"Allocating KV cache with shape: {alloc_shape} consisting of {self.num_blocks} blocks.")
        self._kv_caches = self._init_device_caches()
        self.total_physical_cache_size_in_bytes = (
            self.elem_size_in_bytes
            * self.num_layers
            * 2
            * self.num_blocks
            * self.block_size
            * self.head_num
            * self.head_size
        )
        # Logical cache blocks allocation
        self._available_blocks = self.num_blocks
        self._cache_blocks = tuple(self._init_logical_caches())
        # block availablity state 0->allocated, 1->free
        self._block_states = torch.ones((self.num_blocks,), dtype=torch.bool)
        self._block_states_cum = torch.zeros(size=(self.num_blocks + 1,), dtype=torch.int64)
        self._block_finder = torch.zeros((self.num_blocks,), dtype=torch.int64)

    def get_total_num_blocks(self) -> int:
        """Get the total number of logical cache blocks."""
        return self.num_blocks

    def get_num_available_blocks(self) -> int:
        """Get the number of available cache blocks."""
        return self._available_blocks

    def get_max_blocks_per_sequence(self) -> int:
        """Get the maximum number of blocks that can be allocated for a single sequence."""
        # TODO Consider removing this function as we plan to implement "half-dynamic" batching in schduler/request handler,
        #      which will make the max_blocks_per_sequence dynamic based on the prompt lengths of sequences
        #      in the current batch.
        return self.max_blocks_per_sequence

    def get_block_kv_ptrs(self, block_id: int, layer_id: int) -> Tuple[List[int], List[int]]:
        """Get the key and value pointers of physical caches (of specific layer) corresponding to a logical cache block."""
        block: CacheBlock = self._cache_blocks[block_id]
        return block.k_ptrs[layer_id], block.v_ptrs[layer_id]

    def get_block_table_kv_ptrs(self, block_table: torch.Tensor, layer_id: int) -> Tuple[int, int]:
        """Get the key and value pointers of physical caches (of specific layer) corresponding to logical cache blocks indicated by the block table."""
        k_ptrs = []
        v_ptrs = []
        for block_id in block_table:
            if block_id >= 0:
                block: CacheBlock = self._cache_blocks[block_id]
                k_ptrs.append(block.k_ptrs[layer_id])
                v_ptrs.append(block.v_ptrs[layer_id])
        return k_ptrs, v_ptrs

    def allocate_context_from_block_table(self, block_table: torch.Tensor, context_len: int) -> None:
        """Allocate the logical cache blocks for a single sequence during prefill stage,
        and updates the provided block table with the allocated block ids.

        Args:
            block_table: A 1D tensor of shape [max_blocks_per_sequence], storing mapping of token_position_id -> block_id.
            context_len: The length of the processing sequnece.
        """
        assert block_table.dim() == 1
        if not torch.all(block_table < 0):
            self.logger.error("Some slots on provided block table have been allocated.")
        blocks_required = (context_len + self.block_size - 1) // self.block_size
        if blocks_required > self._available_blocks:
            self.logger.warning(
                f"No enough blocks to allocate. Available blocks {self._available_blocks}; context length {context_len}."
            )
            return

        # Try contiguous allocation
        torch.cumsum(self._block_states, dim=-1, out=self._block_states_cum[1:])
        torch.subtract(
            self._block_states_cum[blocks_required:],
            self._block_states_cum[:-blocks_required],
            out=self._block_finder[blocks_required - 1 :],
        )
        end_indexes = torch.nonzero(self._block_finder == blocks_required, as_tuple=False).view(-1)
        if end_indexes.numel() > 0:
            # contiguous cache exists
            end_idx = end_indexes[0].item() + 1  # open interval
            start_idx = end_idx - blocks_required  # closed interval
            block_indexes = torch.arange(start_idx, end_idx, device=block_table.device)
        else:
            # non-contiguous cache
            available_block_indexes = torch.nonzero(self._block_states == 0).view(-1)
            block_indexes = available_block_indexes[:blocks_required]
        # Update block table
        block_table[:blocks_required] = block_indexes
        # Update cache blocks
        self._block_states[block_indexes] = 0
        self._available_blocks -= blocks_required
        for block_id in block_indexes.tolist():
            block: CacheBlock = self._cache_blocks[block_id]
            block.add_ref()
            if block_id == block_indexes[-1].item():
                self._allocate_on_block(
                    block, block.block_size if context_len % block.block_size == 0 else context_len % block.block_size
                )
            else:
                self._allocate_on_block(block, block.block_size)

    def allocate_token_from_block_table(self, block_table: torch.Tensor, context_len: int) -> None:
        """Allocate the logical cache block for a single sequence during decoding stage,
        and updates the provided block table if a new cache block is needed.

        Args:
            block_table: A 1D tensor of shape [max_blocks_per_sequence], storing mapping of token_position_id -> block_id.
            context_len: The length of the processing sequnece (already-allocated length).
        """
        assert block_table.dim() == 1
        # The last allocated block may be either partially or fully occupied.
        # `alloc_local_block_idx` is the index of block to be allocated on provided block table.
        alloc_local_block_idx = context_len // self.block_size
        self.allocate_single_block(block_table, alloc_local_block_idx, 1)

    def allocate_single_block(self, block_table: torch.Tensor, block_local_idx: int, space_asked: int) -> int:
        """Allocate space asked on a single block in the block table, specified by the provided position id,
        and updates the provided block table with the allocated block.

        Args:
            block_table: A 1D tensor of shape [max_blocks_per_sequence], storing mapping of token_position_id -> block_id.
            block_local_idx: The index of the block in the block table.
            space_asked: i.e. The number of tokens to be assigned space for.
        Returns:
            The remaining space required to be allocated (in other blocks).
        """
        assert block_table.dim() == 1
        block_global_id = block_table[block_local_idx].item()
        if block_global_id < 0:
            # Allocate a new block if the current position is not assigned a block yet
            assert self._available_blocks > 0, "No available blocks to allocate."
            free_block_id = torch.nonzero(self._block_states == 1).view(-1)[0]
            block: CacheBlock = self._cache_blocks[free_block_id]
            block.add_ref()
            block_global_id = block.block_id
            self._available_blocks -= 1
            self._block_states[block_global_id] = 0
            block_table[block_local_idx] = block_global_id
        block: CacheBlock = self._cache_blocks[block_global_id]
        return self._allocate_on_block(block, space_asked)

    def free_block_table(self, block_table: torch.Tensor) -> None:
        """Free the logical cache blocks for **a single sequence**."""
        assert block_table.dim() == 1
        for i in range(block_table.numel()):
            global_block_id = block_table[i].item()
            if global_block_id < 0:
                return
            block: CacheBlock = self._cache_blocks[global_block_id]
            block.remove_ref()
            if not block.has_ref():
                block.allocated_size = 0
                self._available_blocks += 1
                self._block_states[global_block_id] = 1
                # reset the block id in the block table (if we maintain a 2D tensors as block tables in Engine)
                block_table[i] = -1

    def clear_all(self) -> None:
        """Clear all the references and allocations on all the cache blocks."""
        for block in self._cache_blocks:
            block.clear()
        self._available_blocks = self.num_blocks
        self._block_states[:] = 1

    def get_physical_cache(self, layer_id: int, block_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the tensor corresponding to the cache block with the prompted id for a specific layer."""
        return self._kv_caches[0][layer_id][block_idx], self._kv_caches[1][layer_id][block_idx]

    def _allocate_on_block(self, block: CacheBlock, space_asked: int) -> int:
        """Allocate a specific size of space on a provided cache block.

        Returns:
            The remaining space required to be allocated (in other blocks).
        """
        assert block.available_space > 0, "No available space on block to allocate."
        space_to_allocate = min(block.available_space, space_asked)
        block.allocate(space_to_allocate)
        return space_asked - space_to_allocate

    def _init_logical_caches(self):
        """Initialize the logical cache blocks.

        NOTE This function should be called only after the physical caches have been allocated.
        The data pointers of physical caches will be binded to each logical cache block.
        """
        assert self._kv_caches is not None and len(self._kv_caches[0]) > 0
        blocks = []
        physical_block_size = self.elem_size_in_bytes * self.block_size * self.head_num * self.head_size
        k_ptrs = [
            self._kv_caches[0][layer_idx].data_ptr() - physical_block_size for layer_idx in range(self.num_layers)
        ]
        v_ptrs = [
            self._kv_caches[1][layer_idx].data_ptr() - physical_block_size for layer_idx in range(self.num_layers)
        ]
        for i in range(self.num_blocks):
            k_ptrs = [first_block_ptr + physical_block_size for first_block_ptr in k_ptrs]
            v_ptrs = [first_block_ptr + physical_block_size for first_block_ptr in v_ptrs]
            cache_block = CacheBlock(i, self.block_size, self.elem_size_in_bytes, k_ptrs, v_ptrs)
            blocks.append(cache_block)
        return blocks

    def _init_device_caches(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize the physical cache on the device.

        For each layer of the model, we allocate two tensors for key and value respectively,
        with shape of [num_blocks, num_kv_heads, head_size, block_size]
        """
        alloc_shape = (self.num_blocks, self.head_num, self.head_size, self.block_size)
        # TODO: Explore the performance when using difference shapes with kernel-related optimizations
        #       e.g. [num_blocks, num_kv_heads // x, head_size, block_size, x]
        k_cache: List[torch.Tensor] = []
        v_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            k_cache.append(torch.zeros(alloc_shape, dtype=self.dtype, device=self.device))
            v_cache.append(torch.zeros(alloc_shape, dtype=self.dtype, device=self.device))
        return k_cache, v_cache
