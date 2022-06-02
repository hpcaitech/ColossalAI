import torch
import torch.distributed as dist
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Deque, Set, List
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


STATE_TRANS = ((TensorState.FREE, TensorState.HOLD), (TensorState.FREE, TensorState.COMPUTE),
               (TensorState.HOLD, TensorState.FREE), (TensorState.HOLD, TensorState.COMPUTE),
               (TensorState.COMPUTE, TensorState.HOLD), (TensorState.COMPUTE, TensorState.HOLD_AFTER_BWD),
               (TensorState.COMPUTE, TensorState.READY_FOR_REDUCE), (TensorState.HOLD_AFTER_BWD, TensorState.COMPUTE),
               (TensorState.HOLD_AFTER_BWD, TensorState.READY_FOR_REDUCE), (TensorState.READY_FOR_REDUCE,
                                                                            TensorState.HOLD))


@dataclass
class TensorInfo:
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
        self.data = torch.empty(chunk_size, dtype=dtype, device=self.device)
        if not self.is_src_rank:
            self.data.storage().resize_(0)
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
            tensor.data = self.data[self.utilized_size:new_utilized_size].view(tensor.shape)
        else:
            tensor.storage().resize_(0)
        self.tensors_info[tensor] = TensorInfo(tensor_state, self.utilized_size, new_utilized_size)
        self.utilized_size = new_utilized_size

    def release(self) -> None:
        if not self.is_src_rank:
            self.data.storage().resize_(0)
            self._update_tensors_state(TensorState.FREE)

    def _update_tensors_ptr(self) -> None:
        for tensor, tensor_info in self.tensors_info.items():
            tensor.data = self.data[tensor_info.offset:tensor_info.end].view(tensor.shape)

    def _update_tensors_state(self, next_state: TensorState, prev_state: Optional[TensorState] = None):
        for tensor_info in self.tensors_info.values():
            if prev_state is None or tensor_info.state == prev_state:
                tensor_info.state = next_state

    def access(self) -> None:
        if not self.is_src_rank:
            self.data.storage().resize_(self.size)
        self.data.data = self.data.to(get_current_device())
        dist.broadcast(self.data, self.src_rank, group=gpc.get_group(ParallelMode.DATA))
        self._update_tensors_ptr()
        if not self.is_src_rank:
            self._update_tensors_state(TensorState.HOLD, prev_state=TensorState.FREE)

    def move_device(self, device: torch.device) -> None:
        self.data.data = self.data.to(device)
        self._update_tensors_ptr()

    def reduce(self, is_all_reduce: bool = False) -> None:
        self.data.data = self.data.to(get_current_device())
        if is_all_reduce:
            dist.all_reduce(self.data, group=gpc.get_group(ParallelMode.DATA))
        else:
            dist.reduce(self.data, self.src_rank, group=gpc.get_group(ParallelMode.DATA))
        self._update_tensors_ptr()
        self._update_tensors_state(TensorState.HOLD)

    def tensor_trans_state(self, tensor: torch.Tensor, tensor_state: TensorState) -> None:
        assert tensor != TensorState.FREE, 'Can only set a chunk of tensors to FREE'
        # As the gradient hook can be triggered either before or after post-backward
        # tensor's state can be compute -> hold_after_bwd -> ready_for_reduce
        # or compute -> ready_for_reduce -> hold_after_bwd
        # the second one is invalid, we just ignore ready_for_reduce -> hold_after_bwd
        # this function only apply valid state transformation
        # invalid calls will be ignored and nothing changes
        if (self.tensors_info[tensor].state, tensor_state) not in STATE_TRANS:
            # print(
            #     f'WARNING: Rank{gpc.get_global_rank()} apply invalid state trans: {self.tensors_info[tensor].state} to {tensor_state}'
            # )
            return
        self.tensors_info[tensor].state = tensor_state

    def copy_tensor_to_chunk_slice(self, tensor: torch.Tensor, data_slice: torch.Tensor) -> None:
        tensor_info = self.tensors_info[tensor]
        self.data[tensor_info.offset:tensor_info.end].copy_(data_slice.view(-1))
        tensor.data = self.data[tensor_info.offset:tensor_info.end].view(tensor.shape)

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
        return self.data.storage().size() == 0

    def __repr__(self) -> str:
        return f'Chunk: src rank={self.src_rank} ,size={self.size}, utilization={self.utilized_size/self.size*100:.2f}%, freed={self.is_free}, tensor states={[info.state.name for info in self.tensors_info.values()]}'

    @property
    def has_inf_or_nan(self) -> bool:
        return torch.isinf(self.data[:self.utilized_size]).any().item() or \
            torch.isnan(self.data[:self.utilized_size]).any().item()


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
        self.lazy_release_tensors: List[torch.Tensor] = []
        if enable_distributed_storage and chunk_size is None:
            self.rank_load: Dict[str, torch.Tensor] = {}

    def append_tensor(self, tensor: torch.Tensor, group_name: str) -> None:
        assert tensor not in self.tensor_chunk_map
        if self.chunk_size is not None and tensor.numel() > self.chunk_size:
            raise ValueError(
                f'Cannot create chunk, got tensor numel ({tensor.numel()}) > chunk size ({self.chunk_size})')
        if group_name not in self.chunk_groups:
            self.chunk_groups[group_name] = deque()
        try:
            self.chunk_groups[group_name][-1].append(tensor)
        except (IndexError, ChunkFullError):
            chunk_size = self.chunk_size or tensor.numel()
            src_rank = self._get_next_src_rank(group_name)
            chunk = Chunk(chunk_size, src_rank, tensor.dtype, self.device)
            if self.enable_distributed_storage and self.chunk_size is None:
                self.rank_load[group_name][src_rank] += chunk_size
            self.chunk_groups[group_name].append(chunk)
            chunk.append(tensor)
        self.tensor_chunk_map[tensor] = self.chunk_groups[group_name][-1]
        if not self.enable_distributed_storage:
            self.accessed_chunks.add(self.chunk_groups[group_name][-1])

    def _get_next_src_rank(self, group_name: str) -> int:
        if not self.enable_distributed_storage:
            return gpc.get_local_rank(ParallelMode.DATA)
        if self.chunk_size is None:
            if group_name not in self.rank_load:
                self.rank_load[group_name] = torch.zeros(gpc.get_world_size(ParallelMode.DATA), dtype=torch.int64)
            src_rank = torch.argmin(self.rank_load[group_name]).item()
        else:
            chunk_idx = len(self.chunk_groups[group_name])
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
        if chunk not in self.accessed_chunks:
            return
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

    def reduce_chunk(self, tensor: torch.Tensor) -> bool:
        chunk = self.tensor_chunk_map[tensor]
        if not chunk.can_reduce:
            return False
        chunk.reduce(is_all_reduce=not self.enable_distributed_storage)
        return True

    def copy_tensor_to_chunk_slice(self, tensor: torch.Tensor, data: torch.Tensor) -> None:
        chunk = self.tensor_chunk_map[tensor]
        chunk.copy_tensor_to_chunk_slice(tensor, data)

    def is_chunk_free(self, tensor: torch.Tensor) -> bool:
        chunk = self.tensor_chunk_map[tensor]
        return chunk.is_free

    def get_chunk(self, tensor: torch.Tensor) -> Chunk:
        return self.tensor_chunk_map[tensor]

    def add_lazy_release_tensors(self, tensors: List[torch.Tensor]) -> None:
        self.lazy_release_tensors.extend(tensors)

    def exec_lazy_release(self) -> None:
        for tensor in self.lazy_release_tensors:
            self.release_chunk(tensor)
        self.lazy_release_tensors.clear()

    def __repr__(self) -> str:
        msg = f'Rank {gpc.get_local_rank(ParallelMode.DATA)}:\n'
        for group_name, group in self.chunk_groups.items():
            msg += f'Group {group_name}:\n'
            for i, chunk in enumerate(group):
                msg += f'[{i}] {chunk}\n'
        return msg
