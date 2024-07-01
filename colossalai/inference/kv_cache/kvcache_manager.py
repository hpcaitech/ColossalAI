from typing import List, Tuple

import torch
from transformers.configuration_utils import PretrainedConfig

from colossalai.inference.config import InferenceConfig
from colossalai.inference.struct import Sequence
from colossalai.logging import get_dist_logger
from colossalai.utils import get_current_device

from .block_cache import CacheBlock

__all__ = ["KVCacheManager"]

GIGABYTE = 1024**3


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
        and the physical caches, each with size of `block_size * kv_head_num * head_size * elem_size` for a single layer,
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

    def __init__(self, config: InferenceConfig, model_config: PretrainedConfig) -> None:
        self.logger = get_dist_logger(__name__)
        self.device = get_current_device()

        # Parallel settings
        self.tp_size = config.tp_size
        # Model settings
        self.dtype = config.dtype

        if config.kv_cache_dtype is None:
            self.kv_cache_dtype = config.dtype
        else:
            self.kv_cache_dtype = config.kv_cache_dtype

        self.elem_size_in_bytes = torch.tensor([], dtype=self.dtype).element_size()
        self.num_layers = model_config.num_hidden_layers
        self.head_num = model_config.num_attention_heads
        self.head_size = model_config.hidden_size // self.head_num
        if hasattr(model_config, "num_key_value_heads"):
            self.kv_head_num = model_config.num_key_value_heads
        else:
            self.kv_head_num = self.head_num

        assert (
            self.kv_head_num % self.tp_size == 0
        ), f"Cannot shard {self.kv_head_num} heads with tp size {self.tp_size}"
        self.kv_head_num //= self.tp_size
        self.beam_width = config.beam_width
        self.max_batch_size = config.max_batch_size
        self.max_input_length = config.max_input_len
        self.max_output_length = config.max_output_len
        # Cache block settings
        self.block_size = config.block_size

        # NOTE: `num_blocks` is not prompted, but evaluated from the maximum input/output length, and the maximum batch size
        if config.enable_streamingllm:
            self.max_blocks_per_sequence = (
                config.start_token_size + config.generated_token_size + self.block_size - 1
            ) // self.block_size + 1
        else:
            self.max_blocks_per_sequence = (
                self.max_input_length + self.max_output_length + self.block_size - 1
            ) // self.block_size
        self.num_blocks = self.max_blocks_per_sequence * self.max_batch_size * self.beam_width

        # Physical cache allocation
        if config.use_cuda_kernel:
            x = 16 // torch.tensor([], dtype=config.dtype).element_size()
            kalloc_shape = (self.num_blocks, self.kv_head_num, self.head_size // x, self.block_size, x)
            valloc_shape = (self.num_blocks, self.kv_head_num, self.block_size, self.head_size)
            self.logger.info(
                f"Allocating K cache with shape: {kalloc_shape}, V cache with shape: {valloc_shape} consisting of {self.num_blocks} blocks."
            )
            self._kv_caches = self._init_device_caches(kalloc_shape, valloc_shape)
        else:
            alloc_shape = (self.num_blocks, self.kv_head_num, self.block_size, self.head_size)
            self.logger.info(f"Allocating KV cache with shape: {alloc_shape} consisting of {self.num_blocks} blocks.")
            self._kv_caches = self._init_device_caches(alloc_shape, alloc_shape)
        self.total_physical_cache_size_in_bytes = (
            self.elem_size_in_bytes
            * self.num_layers
            * 2
            * self.num_blocks
            * self.block_size
            * self.kv_head_num
            * self.head_size
        )
        self.logger.info(
            f"Allocated {self.total_physical_cache_size_in_bytes / GIGABYTE:.2f} GB of KV cache on device {self.device}."
        )
        # Logical cache blocks allocation
        self._available_blocks = self.num_blocks
        self._cache_blocks = tuple(self._init_logical_caches())
        # block availablity state 0->allocated, 1->free
        self._block_states = torch.ones((self.num_blocks,), dtype=torch.bool)
        self._block_states_cum = torch.zeros(size=(self.num_blocks + 1,), dtype=torch.int64)
        self._block_finder = torch.zeros((self.num_blocks,), dtype=torch.int64)

    @property
    def total_num_blocks(self) -> int:
        """Get the total number of logical cache blocks."""
        return self.num_blocks

    @property
    def num_available_blocks(self) -> int:
        """Get the number of available cache blocks."""
        return self._available_blocks

    def get_head_size(self):
        return self.head_size

    def get_kv_cache(self):
        """Get k_cache and v_cache"""
        return self._kv_caches

    def get_max_blocks_per_sequence(self) -> int:
        """Get the maximum number of blocks that can be allocated for a single sequence."""
        # TODO Consider removing this function as we plan to implement "half-dynamic" batching in schduler/request handler,
        #      which will make the max_blocks_per_sequence dynamic based on the prompt lengths of sequences
        #      in the current batch.
        return self.max_blocks_per_sequence

    def check_allocation(self, seq: Sequence) -> bool:
        num_blocks_needed = (seq.input_len + self.max_output_length + self.block_size - 1) // self.block_size
        return num_blocks_needed <= self.num_available_blocks

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
            block_table: A 1D tensor of shape [max_blocks_per_sequence], mapping of token_position_id -> block_id.
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

    def allocate_context_from_block_tables(self, block_tables: torch.Tensor, context_lengths: torch.Tensor) -> None:
        """Allocate logical cache blocks for a batch of sequences during prefill stage.

        Args:
            block_tables (torch.Tensor): [bsz, max_blocks_per_sequence]
            context_lengths (torch.Tensor): [bsz]]
        """
        assert block_tables.dim() == 2
        assert block_tables.size(0) == context_lengths.size(0)
        if not torch.all(block_tables < 0):
            self.logger.error("Some slots on provided block table have been allocated.")
        blocks_required = (context_lengths + self.block_size - 1) // self.block_size
        num_blocks_required = torch.sum(blocks_required).item()
        assert isinstance(num_blocks_required, int)
        if num_blocks_required > self._available_blocks:
            self.logger.warning(
                f"Lacking blocks to allocate. Available blocks {self._available_blocks}; blocks asked {num_blocks_required}."
            )
            return

        bsz = block_tables.size(0)
        # Try contiguous allocation
        torch.cumsum(self._block_states, dim=-1, out=self._block_states_cum[1:])
        torch.subtract(
            self._block_states_cum[num_blocks_required:],
            self._block_states_cum[:-num_blocks_required],
            out=self._block_finder[num_blocks_required - 1 :],
        )
        end_indexes = torch.nonzero(self._block_finder == num_blocks_required, as_tuple=False).view(-1)
        if end_indexes.numel() > 0:
            # contiguous cache exists
            end_idx = end_indexes[0].item() + 1  # open interval
            start_idx = end_idx - num_blocks_required  # closed interval
            alloc_block_ids = torch.arange(start_idx, end_idx)
            for i in range(bsz):
                curr_required = blocks_required[i]
                block_tables[i, :curr_required] = torch.arange(
                    start_idx, start_idx + curr_required, device=block_tables.device
                )
                start_idx += curr_required
        else:
            # non-contiguous cache
            available_block_ids = torch.nonzero(self._block_states > 0).view(-1)
            alloc_block_ids = available_block_ids[:num_blocks_required]
            alloc_block_ids = alloc_block_ids.to(dtype=block_tables.dtype, device=block_tables.device)
            start_idx = 0
            for i in range(bsz):
                curr_required = blocks_required[i]
                block_tables[i, :curr_required] = alloc_block_ids[start_idx, start_idx + curr_required]
                start_idx += curr_required

        # Update cache blocks
        self._block_states[alloc_block_ids] = 0
        self._available_blocks -= num_blocks_required
        last_block_locs = torch.cumsum(blocks_required, dim=0) - 1
        last_block_locs = last_block_locs.to(device=alloc_block_ids.device)

        for i, block_id in enumerate(alloc_block_ids[last_block_locs]):
            block: CacheBlock = self._cache_blocks[block_id]
            block.add_ref()
            self._allocate_on_block(
                block,
                (
                    block.block_size
                    if context_lengths[i] % block.block_size == 0
                    else context_lengths[i].item() % block.block_size
                ),
            )
        for block_id in alloc_block_ids:
            if block_id in alloc_block_ids[last_block_locs]:
                continue
            block: CacheBlock = self._cache_blocks[block_id]
            block.add_ref()
            self._allocate_on_block(block, block.block_size)

    def allocate_token_from_block_table(self, block_table: torch.Tensor, context_len: int) -> None:
        """Allocate the logical cache block for a single sequence during decoding stage,
        and updates the provided block table if a new cache block is needed.

        Args:
            block_table: A 1D tensor of shape [max_blocks_per_sequence], mapping of token_position_id -> block_id.
            context_len: The length of the processing sequnece (already-allocated length).
        """
        assert block_table.dim() == 1
        # The last allocated block may be either partially or fully occupied.
        # `alloc_local_block_idx` is the index of block to be allocated on provided block table.
        alloc_local_block_idx = context_len // self.block_size
        return self.allocate_single_block(block_table, alloc_local_block_idx)

    def allocate_tokens_from_block_tables(
        self, block_tables: torch.Tensor, context_lens: torch.Tensor, bsz: int = None
    ) -> List[int]:
        """Allocate logical cache blocks for a batch of sequences during decoding stage.

        Usage:
            allocate_context_from_block_tables
            model forward (block tables & context lengths passed)
            update context lengths
            allocate_tokens_from_block_tables
            model forward
            update context lengths
            allocate_tokens_from_block_tables
            model forward
            update context lengths
            ...

        Args:
            block_tables (torch.Tensor): [bsz, max_blocks_per_sequence]
            context_lengths (torch.Tensor): [bsz]

        Returns:
            List[int]: list of sequence uid to be recycled
        """
        assert block_tables.dim() == 2
        assert context_lens.dim() == 1

        bsz = block_tables.size(0) if bsz is None else bsz

        alloc_local_block_indexes = (context_lens[:bsz]) // self.block_size
        block_global_ids = block_tables[torch.arange(0, bsz), alloc_local_block_indexes]
        seqs_to_recycle = []
        new_blocks_required = torch.sum(block_global_ids < 0).item()
        seqs_req_new_blocks = torch.nonzero(block_global_ids < 0).squeeze()

        if new_blocks_required > 0:
            if new_blocks_required > self._available_blocks:
                # TODO might want to revise the logic here
                # Process the first (_available_blocks) sequences that require new blocks
                # Put the rest of the sequences back to recycled
                seqs_req_new_blocks, seqs_to_recycle = (
                    seqs_req_new_blocks[: self._available_blocks],
                    seqs_req_new_blocks[self._available_blocks :],
                )
                for seq_id in seqs_to_recycle:
                    self.free_block_table(block_tables[seq_id])
                new_blocks_required = self._available_blocks

            # NOTE might want to alloc contiguous logic
            free_block_ids = torch.nonzero(self._block_states > 0).view(-1)
            alloc_block_ids = free_block_ids[:new_blocks_required].to(
                dtype=block_tables.dtype, device=block_tables.device
            )

            for block_id in alloc_block_ids:
                block: CacheBlock = self._cache_blocks[block_id]
                block.add_ref()
                self._block_states[block_id] = 0
                self._available_blocks -= 1
            block_tables[seqs_req_new_blocks, alloc_local_block_indexes[seqs_req_new_blocks]] = alloc_block_ids
            block_global_ids = block_tables[torch.arange(0, bsz), alloc_local_block_indexes]

        for block_id in block_global_ids:
            self._allocate_on_block(self._cache_blocks[block_id], 1)

        return seqs_to_recycle

    def allocate_n_tokens_from_block_tables(
        self,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        bsz: int,
        n: int,
    ) -> List[int]:
        """Allocate logical cache blocks for `n` new tokens for a batch of sequences during decoding stage."""
        assert block_tables.dim() == 2
        assert context_lens.dim() == 1

        bsz = block_tables.size(0) if bsz is None else bsz
        assert bsz == 1, "Support bsz 1 for now"  # TODO support bsz > 1

        seqs_to_recycle = []
        for i in range(n):
            seqs_to_recycle += self.allocate_tokens_from_block_tables(block_tables, context_lens - n + i + 1, bsz)

        return seqs_to_recycle

    def allocate_single_block(self, block_table: torch.Tensor, block_local_idx: int) -> int:
        """Allocate space asked on a single block in the block table, specified by the provided position id,
        and updates the provided block table with the allocated block.

        Args:
            block_table: A 1D tensor of shape [max_blocks_per_sequence], mapping of token_position_id -> block_id.
            block_local_idx: The index of the block in the block table.
            space_asked: i.e. The number of tokens to be assigned space for.
        Returns:
            The remaining space required to be allocated (in other blocks).
        """
        space_asked = 1
        block_global_id = block_table[block_local_idx].item()
        if block_global_id < 0:
            # Allocate a new block if the current position is not assigned a block yet
            if self._available_blocks <= 0:
                # No available blocks to allocate, we free current sequence and return it to
                self.free_block_table(block_table)
                return True
            free_block_id = torch.nonzero(self._block_states == 1).view(-1)[0]
            block: CacheBlock = self._cache_blocks[free_block_id]
            block.add_ref()
            block_global_id = block.block_id
            self._available_blocks -= 1
            self._block_states[block_global_id] = 0
            block_table[block_local_idx] = block_global_id
        block: CacheBlock = self._cache_blocks[block_global_id]
        return self._allocate_on_block(block, space_asked)
        # only when space asked if fully satisfied, the return value will be zero.

    def free_block_table(self, block_table: torch.Tensor) -> None:
        """Free the logical cache blocks for **a single sequence**."""
        assert block_table.dim() == 1
        for i, global_block_id in enumerate(block_table.tolist()):
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

    def free_block_tables(self, block_tables: torch.Tensor, first_n: int = None) -> None:
        """Release the logical cache blocks for a batch of sequences.
        If `first_n` is provided, only the blocks for the first several sequences will be released.
        """
        assert block_tables.dim() == 2
        first_n = block_tables.size(0) if first_n is None else first_n
        for block_table in block_tables[:first_n]:
            self.free_block_table(block_table)

    def clear_all(self) -> None:
        """Clear all the references and allocations on all the cache blocks."""
        for block in self._cache_blocks:
            block.clear()
        self._available_blocks = self.num_blocks
        self._block_states[:] = 1

    def streamingllm_free_block_tables(self, updated_block_ids: List[int]):
        """
        Free the block that needs to be swapped out.
        """
        for global_block_id in updated_block_ids:
            if global_block_id < 0:
                return
            block: CacheBlock = self._cache_blocks[global_block_id]
            block.remove_ref()
            if not block.has_ref():
                block.allocated_size = 0
                self._available_blocks += 1
                self._block_states[global_block_id] = 1

    def get_physical_cache(self, layer_id: int, block_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the tensor corresponding to the cache block with the prompted id for a specific layer."""
        return self._kv_caches[0][layer_id][block_idx], self._kv_caches[1][layer_id][block_idx]

    def _allocate_on_block(self, block: CacheBlock, space_asked: int) -> int:
        """Allocate a specific size of space on a provided cache block.

        Returns:
            The remaining space required to be allocated (in other blocks).
        """
        assert block.available_space > 0, f"Found no available space left in the chosen block {block}."
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
        physical_block_size = self.elem_size_in_bytes * self.block_size * self.kv_head_num * self.head_size
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

    def _init_device_caches(
        self, kalloc_shape: Tuple[int, ...], valloc_shape: Tuple[int, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize the physical cache on the device.

        For each layer of the model, we allocate two tensors for key and value respectively,
        with shape of [num_blocks, num_kv_heads, block_size, head_size]
        """
        k_cache: List[torch.Tensor] = []
        v_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            k_cache.append(torch.zeros(kalloc_shape, dtype=self.kv_cache_dtype, device=self.device))
            v_cache.append(torch.zeros(valloc_shape, dtype=self.kv_cache_dtype, device=self.device))
        return k_cache, v_cache


class RPCKVCacheManager(KVCacheManager):
    def __init__(self, config: InferenceConfig, model_config: PretrainedConfig, verbose: bool = False) -> None:
        self.logger = get_dist_logger(__name__)
        self.device = get_current_device()
        self.config = config

        # Parallel settings
        self.tp_size = config.tp_size
        # Model settings
        self.dtype = config.dtype
        self.elem_size_in_bytes = torch.tensor([], dtype=self.dtype).element_size()
        self.num_layers = model_config.num_hidden_layers
        self.head_num = model_config.num_attention_heads
        self.head_size = model_config.hidden_size // self.head_num
        if hasattr(model_config, "num_key_value_heads"):
            self.kv_head_num = model_config.num_key_value_heads
        else:
            self.kv_head_num = self.head_num

        if config.kv_cache_dtype is None:
            self.kv_cache_dtype = config.dtype
        else:
            self.kv_cache_dtype = config.kv_cache_dtype

        assert (
            self.kv_head_num % self.tp_size == 0
        ), f"Cannot shard {self.kv_head_num} heads with tp size {self.tp_size}"
        self.kv_head_num //= self.tp_size
        self.beam_width = config.beam_width
        self.max_batch_size = config.max_batch_size
        self.max_input_length = config.max_input_len
        self.max_output_length = config.max_output_len
        # Cache block settings
        self.block_size = config.block_size

        # NOTE: `num_blocks` is not prompted, but evaluated from the maximum input/output length, and the maximum batch size
        if config.enable_streamingllm:
            self.max_blocks_per_sequence = (
                config.start_token_size + config.generated_token_size + self.block_size - 1
            ) // self.block_size + 1
        else:
            self.max_blocks_per_sequence = (
                self.max_input_length + self.max_output_length + self.block_size - 1
            ) // self.block_size
        self.num_blocks = self.max_blocks_per_sequence * self.max_batch_size * self.beam_width

        # Logical cache blocks allocation
        self._available_blocks = self.num_blocks
        self._cache_blocks = tuple(self._init_logical_caches())
        # block availablity state 0->allocated, 1->free
        self._block_states = torch.ones((self.num_blocks,), dtype=torch.bool)
        self._block_states_cum = torch.zeros(size=(self.num_blocks + 1,), dtype=torch.int64)
        self._block_finder = torch.zeros((self.num_blocks,), dtype=torch.int64)

    def get_physical_cache_shape(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        # Physical cache allocation
        if self.config.use_cuda_kernel:
            x = 16 // torch.tensor([], dtype=self.config.dtype).element_size()
            kalloc_shape = (self.num_blocks, self.kv_head_num, self.head_size // x, self.block_size, x)
            valloc_shape = (self.num_blocks, self.kv_head_num, self.block_size, self.head_size)
            self.logger.info(
                f"Allocating K cache with shape: {kalloc_shape}, V cache with shape: {valloc_shape} consisting of {self.num_blocks} blocks."
            )
        else:
            alloc_shape = (self.num_blocks, self.kv_head_num, self.block_size, self.head_size)
            kalloc_shape = alloc_shape
            valloc_shape = alloc_shape
            self.logger.info(f"Allocating KV cache with shape: {alloc_shape} consisting of {self.num_blocks} blocks.")
        return kalloc_shape, valloc_shape

    def get_kv_cache(self):
        """Get k_cache and v_cache"""
        return NotImplementedError

    def _init_logical_caches(self):
        """Initialize the logical cache blocks."""
        blocks = []
        for i in range(self.num_blocks):
            cache_block = CacheBlock(i, self.block_size, self.elem_size_in_bytes, k_ptrs=None, v_ptrs=None)
            blocks.append(cache_block)
        return blocks
