import torch
import torch.distributed as dist
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Deque, Set
from collections import deque
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.utils import get_current_device


class TensorState(Enum):
    FREE = 0
    COMPUTE = 1
    HOLD = 2
    HOLD_AFTER_BWD = 3
    READY_FOR_REDUCE = 4


class TensorInfo(dataclass):
    state: TensorState
    offset: int
    end: int


class ChunkFullError(Exception):
    pass


class Chunk:

    def __init__(self,
                 chunk_size: int,
                 src_rank: int,
                 dtype: torch.dtype,
                 init_device: Optional[torch.device] = None) -> None:
        self.size = chunk_size
        self.utilized_size = 0
        self.src_rank = src_rank
        self.is_src_rank = gpc.get_local_rank(ParallelMode.DATA) == src_rank
        self.dtype = dtype
        self.device = init_device or get_current_device()
        init_size = chunk_size if self.is_src_rank else 0
        self.data = torch.empty(init_size, dtype=dtype, device=self.device)
        self.tensors_info: Dict[torch.Tensor, TensorInfo] = {}

    def append(self, tensor: torch.Tensor) -> None:
        assert tensor.dtype == self.dtype
        new_utilized_size = self.utilized_size + tensor.numel()
        if new_utilized_size > self.size:
            raise ChunkFullError
        tensor_state = TensorState.FREE
        if self.is_src_rank:
            self.data[self.utilized_size:new_utilized_size].copy_(tensor.view(-1))
            tensor_state = TensorState.HOLD
            tensor.data = self.data[self.utilized_size:new_utilized_size].view_as(tensor)
        else:
            tensor.storage().resize_(0)
        self.tensors_info[tensor] = TensorInfo(tensor_state, self.utilized_size, new_utilized_size)
        self.utilized_size = new_utilized_size

    def release(self) -> None:
        if not self.is_src_rank:
            self.data = torch.empty(0, dtype=self.dtype, device=self.device)
            for tensor, tensor_info in self.tensors_info.items():
                tensor_info.state = TensorState.FREE
                tensor.storage().resize_(0)

    def _update_tensors_ptr(self) -> None:
        for tensor, tensor_info in self.tensors_info.items():
            tensor.data = self.data[tensor_info.offset:tensor_info.end].view_as(tensor)

    def access(self) -> None:
        if not self.is_src_rank:
            self.data = torch.empty(self.size, dtype=self.dtype, device=get_current_device())
        else:
            self.data = self.data.to(get_current_device())
        dist.broadcast(self.data, self.src_rank, group=gpc.get_group(ParallelMode.DATA))
        if not self.is_src_rank:
            self._update_tensors_ptr()

    def move_device(self, device: torch.device) -> None:
        self.data = self.data.to(device)
        self._update_tensors_ptr()

    def reduce(self) -> None:
        self.data = self.data.to(get_current_device())
        dist.reduce(self.data, self.src_rank, group=gpc.get_group(ParallelMode.DATA))
        self._update_tensors_ptr()
        for tensor_info in self.tensors_info.values():
            tensor_info.state = TensorState.HOLD

    def all_reduce(self) -> None:
        self.data = self.data.to(get_current_device())
        dist.all_reduce(self.data, group=gpc.get_group(ParallelMode.DATA))
        self._update_tensors_ptr()
        for tensor_info in self.tensors_info.values():
            tensor_info.state = TensorState.HOLD

    def tensor_trans_state(self, tensor: torch.Tensor, tensor_state: TensorState) -> None:
        assert tensor != TensorState.FREE, 'Can only set a chunk of tesors to FREE'
        self.tensors_info[tensor].state = tensor_state

    def update_tensor(self, tensor: torch.Tensor, data_slice: torch.Tensor) -> None:
        tensor_info = self.tensors_info[tensor]
        self.data[tensor_info.offset:tensor_info.end].copy_(data_slice.view(-1))
        tensor.storage().resize_(0)
        tensor.data = self.data[tensor_info.offset:tensor_info.end].view_as(tensor)

    @property
    def can_release(self) -> bool:
        for tensor_info in self.tensors_info.values():
            if tensor_info.state != TensorState.HOLD:
                return False
        return True

    @property
    def can_move_device(self) -> bool:
        for tensor_info in self.tensors_info.values():
            if tensor_info.state in (TensorState.COMPUTE, TensorState.READY_FOR_REDUCE):
                return False
        return True

    @property
    def can_reduce(self) -> bool:
        for tensor_info in self.tensors_info.values():
            if tensor_info.state != TensorState.READY_FOR_REDUCE:
                return False
        return True

    @property
    def is_free(self) -> bool:
        return self.data.numel() == 0


class ChunkManager:

    def __init__(self,
                 chunk_size: Optional[int],
                 enable_distributed_storage: bool = False,
                 init_device: Optional[torch.device] = None) -> None:
        assert chunk_size is None or chunk_size > 0
        self.chunk_size = chunk_size
        self.enable_distributed_storage = enable_distributed_storage
        self.device = init_device or get_current_device()
        self.chunk_groups: Dict[str, Deque[Chunk]] = {}
        self.tensor_chunk_map: Dict[torch.Tensor, Chunk] = {}
        self.accessed_chunks: Set[Chunk] = set()

    def append_tensor(self, tensor: torch.Tensor, group_name: str) -> None:
        if self.chunk_size is not None and tensor.numel() > self.chunk_size:
            raise ValueError(
                f'Cannot create chunk, got tensor numel ({tensor.numel()}) > chunk size ({self.chunk_size})')
        if group_name not in self.chunk_groups:
            self.chunk_groups[group_name] = deque()
        try:
            self.chunk_groups[group_name][-1].append(tensor)
        except IndexError or ChunkFullError:
            chunk_size = self.chunk_size or tensor.numel()
            chunk = Chunk(chunk_size, self._get_next_src_rank(group_name), tensor.dtype, self.device)
            self.chunk_groups[group_name].append(chunk)
            chunk.append(tensor)
        self.tensor_chunk_map[tensor] = self.chunk_groups[group_name][-1]
        if not self.enable_distributed_storage:
            self.accessed_chunks.add(self.chunk_groups[group_name][-1])

    def _get_next_src_rank(self, group_name: str) -> int:
        if not self.enable_distributed_storage:
            return gpc.get_local_rank(ParallelMode.DATA)
        chunk_idx = len(self.chunk_groups[group_name])
        if self.chunk_size is None:
            # Don't use chunk, round-robin
            pass
        else:
            src_rank = chunk_idx % gpc.get_world_size(ParallelMode.DATA)
        return src_rank

    def access_chunk(self, tensor: torch.Tensor) -> None:
        chunk = self.tensor_chunk_map[tensor]
        if chunk in self.accessed_chunks:
            return
        chunk.access()
        self.accessed_chunks.add(chunk)

    def release_chunk(self, tensor: torch.Tensor) -> None:
        if not self.enable_distributed_storage:
            return
        chunk = self.tensor_chunk_map[tensor]
        if chunk.can_release:
            chunk.release()
        self.accessed_chunks.remove(chunk)

    def move_chunk(self, tensor: torch.Tensor, device: torch.device) -> None:
        chunk = self.tensor_chunk_map[tensor]
        if chunk.can_move_device:
            chunk.move_device(device)

    def trans_tensor_state(self, tensor: torch.Tensor, state: TensorState) -> None:
        chunk = self.tensor_chunk_map[tensor]
        chunk.tensor_trans_state(tensor, state)

    def reduce_chunk(self, tensor: torch.Tensor) -> None:
        chunk = self.tensor_chunk_map[tensor]
        if not chunk.can_reduce:
            return
        if self.enable_distributed_storage:
            chunk.reduce()
        else:
            chunk.all_reduce()

    def update_tensor(self, tensor: torch.Tensor, data: torch.Tensor) -> None:
        chunk = self.tensor_chunk_map[tensor]
        chunk.update_tensor(tensor, data)

    def is_chunk_free(self, tensor: torch.Tensor) -> bool:
        chunk = self.tensor_chunk_map[tensor]
        return chunk.is_free
