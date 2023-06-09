from typing import Dict, List, Optional, Set

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .chunk import Chunk
from .memory_pool import MemoryPool, TensorBlock
from .states import TensorState


class ChunkGroup(object):
    """ChunkGroup manages a group of chunks and their memory pool.
    Commonly, one model has one chunk group.
    It supports chunk allocation, chunk access, and chunk release.
    ChunkGroup is responsible for the memory management before its APIs.

    args:
        rcache: A memory pool to instantiate chunks.
    """

    def __init__(self, rcache: MemoryPool) -> None:
        super().__init__()
        self.rcache = rcache
        self.fused_chunks: Set[Chunk] = set()
        self.float_chunks: Set[Chunk] = set()
        self.ten_to_chunk: Dict[torch.Tensor, Chunk] = dict()

        self.accessed_fused_chunks: Set[Chunk] = set()
        self.accessed_float_chunks: Set[Chunk] = set()

    def __add_to_accset(self, chunk: Chunk):
        if chunk.rcache_fused:
            self.accessed_fused_chunks.add(chunk)
        else:
            self.accessed_float_chunks.add(chunk)

    def __remove_from_accset(self, chunk: Chunk):
        if chunk.rcache_fused:
            self.accessed_fused_chunks.remove(chunk)
        else:
            self.accessed_float_chunks.remove(chunk)

    def __check_new_float_chunk(self, size: int, dtype: torch.dtype):
        # if the public space is 0, there is no access operations
        if self.rcache.public_space == 0:
            return
        # otherwise, check its size and dtype
        assert size == self.rcache.public_block_size
        assert dtype == self.rcache.public_dtype

    def inside_check(self, chunk: Chunk) -> None:
        """Check whether the chunk is in this ChunkGroup"""
        if chunk.rcache_fused:
            assert chunk in self.fused_chunks
        else:
            assert chunk in self.float_chunks

    def is_accessed(self, chunk: Chunk) -> bool:
        """Chech whether the chunk is accessed."""
        # sanity check
        self.inside_check(chunk)

        if chunk.rcache_fused:
            return (chunk in self.accessed_fused_chunks)
        else:
            return (chunk in self.accessed_float_chunks)

    def open_chunk(self,
                   chunk_size: int,
                   chunk_dtype: torch.dtype,
                   process_group: ProcessGroup,
                   chunk_config: Optional[Dict] = None) -> Chunk:
        """Open a chunk to store parameters."""
        if chunk_config is None:
            chunk_config = {}

        chunk = Chunk(rcache=self.rcache,
                      chunk_size=chunk_size,
                      chunk_dtype=chunk_dtype,
                      process_group=process_group,
                      **chunk_config)
        # sanity check
        if not chunk.rcache_fused:
            self.__check_new_float_chunk(chunk_size, chunk_dtype)

        return chunk

    def close_chunk(self, chunk: Chunk) -> bool:
        """Close the chunk during the allocation."""
        chunk.close_chunk()
        # add the new chunk to the set of allocated chunks
        if chunk.rcache_fused:
            self.fused_chunks.add(chunk)
        else:
            self.float_chunks.add(chunk)
        # add the new chunk to the mapping
        for t in chunk.get_tensors():
            assert t not in self.ten_to_chunk
            self.ten_to_chunk[t] = chunk
        return True

    def allocate_chunk(self,
                       tensor_list: List[torch.Tensor],
                       chunk_size: int,
                       chunk_dtype: torch.dtype,
                       process_group: ProcessGroup,
                       chunk_config: Optional[Dict] = None) -> Chunk:
        """Allocate a chunk for a list of parameters."""
        chunk = self.open_chunk(chunk_size, chunk_dtype, process_group, chunk_config)
        # append tensors
        for t in tensor_list:
            chunk.append_tensor(t)
        self.close_chunk(chunk)

        return chunk

    def tensors_to_chunks(self, tensor_list: List[torch.Tensor]) -> List[Chunk]:
        """Get the chunks of a gevien list of tensors."""
        chunk_list = list()
        for tensor in tensor_list:
            chunk = self.ten_to_chunk.get(tensor)
            if chunk not in chunk_list:
                chunk_list.append(chunk)
        chunk_list.sort(key=lambda c: c.chunk_id)
        return chunk_list

    def rcache_enough_check(self, chunk: Chunk) -> bool:
        """Check whether the rcache has enough blocks to store the gathered chunk."""
        if chunk.rcache_fused:
            return True
        return self.rcache.public_free_count > 0

    def access_chunk(self, chunk: Chunk) -> bool:
        """Access a chunk into rCache."""
        self.inside_check(chunk)
        # if this chunk is accessed already, return False
        if self.is_accessed(chunk):
            return False

        if chunk.rcache_fused:
            block = None
        else:
            block = self.rcache.pop_public_block()
        chunk.access_chunk(block)
        self.__add_to_accset(chunk)
        return True

    def release_chunk(self, chunk: Chunk) -> bool:
        """Release a chunk from rCache."""
        self.inside_check(chunk)
        assert self.is_accessed(chunk)
        assert chunk.scatter_check

        block = chunk.release_chunk()
        if block:
            self.rcache.free_public_block(block)
        self.__remove_from_accset(chunk)
        return True

    def reduce_chunk(self, chunk: Chunk, always_fp32: bool = False, sync: bool = True) -> Optional[TensorBlock]:
        """Reduce and scatter a gradient chunk from rCache."""
        self.inside_check(chunk)
        assert self.is_accessed(chunk)
        assert chunk.reduce_check

        block = chunk.reduce_chunk(always_fp32=always_fp32, sync=sync)
        if block and sync:
            # if synchronized, free the block into rcache
            self.rcache.free_public_block(block)
            block = None

        self.__remove_from_accset(chunk)

        return block

    def tensor_trans_state(self, tensor: torch.Tensor, state: TensorState):
        """Transform the state of a tensor."""
        chunk = self.ten_to_chunk.get(tensor)
        chunk.tensor_trans_state(tensor, state)
