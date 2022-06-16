import torch
import torch.distributed as dist
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Deque, Set, List, Tuple, Iterable
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


def is_storage_empty(tensor: torch.Tensor) -> bool:
    return tensor.storage().size() == 0


def free_storage(tensor: torch.Tensor) -> None:
    if not is_storage_empty(tensor):
        tensor.storage().resize_(0)


def alloc_storage(tensor: torch.Tensor) -> None:
    if is_storage_empty(tensor):
        tensor.storage().resize_(tensor.numel())


class Chunk:
    """
    A chunk is a contiguous memory space which contains multiple tensors.

    Args:
        chunk_size (int): the number of elements in a chunk
        src_rank (int): the process which owns the chunk
        dtype (torch.dtype): the data type of the chunk
        init_device (torch.device): optional, the device where the tensor is initialized. The default value is None, which is the current GPU.
        force_data_on_cuda (bool): optional, if True, chunk.data is always on cuda. Defaults to False.
    """

    def __init__(self,
                 chunk_size: int,
                 src_rank: int,
                 dtype: torch.dtype,
                 init_device: Optional[torch.device] = None,
                 force_data_on_cuda: bool = False) -> None:
        self.size = chunk_size
        self.utilized_size = 0
        self.src_rank = src_rank
        self.is_src_rank = gpc.get_local_rank(ParallelMode.DATA) == src_rank
        self.global_src_rank = gpc.get_ranks_in_group(ParallelMode.DATA)[src_rank]
        self.dtype = dtype
        device = init_device or get_current_device()
        if force_data_on_cuda:
            self.data = torch.empty(chunk_size, dtype=dtype, device=get_current_device())
            self._cpu_data = torch.empty(chunk_size, dtype=dtype)
            if device.type == 'cuda':
                free_storage(self._cpu_data)
            else:
                free_storage(self.data)
        else:
            self.data = torch.empty(chunk_size, dtype=dtype, device=device)
            self._cpu_data = None

        # we only keep the chunk in full in the process by which the tensor is owned
        if not self.is_src_rank:
            free_storage(self._payload)

        # each tensor is associated with a TensorInfo to track meta info
        self.tensors_info: Dict[torch.Tensor, TensorInfo] = {}
        self.mem = self.size * self.data.element_size()

    def append(self, tensor: torch.Tensor) -> None:
        """
        Add a tensor to the chunk.

        Args:
            tensor (torch.Tensor): a tensor to be added to the chunk
        """
        assert tensor.dtype == self.dtype
        new_utilized_size = self.utilized_size + tensor.numel()

        # raise exception when the chunk size is exceeded
        if new_utilized_size > self.size:
            raise ChunkFullError

        # set tensor state
        tensor_state = TensorState.FREE

        # if the process owns the rank, then copy the tensor to its chunk buffer
        # otherwise set its storage size to 0 to reduce memory consumption
        if self.is_src_rank:
            self._payload[self.utilized_size:new_utilized_size].copy_(tensor.view(-1))
            tensor_state = TensorState.HOLD
            tensor.data = self._payload[self.utilized_size:new_utilized_size].view(tensor.shape)
        else:
            tensor.storage().resize_(0)
        self.tensors_info[tensor] = TensorInfo(tensor_state, self.utilized_size, new_utilized_size)
        self.utilized_size = new_utilized_size

    def release(self) -> None:
        """
        Release the memory space on processes which do not own the chunk.
        """
        if not self.is_src_rank:
            free_storage(self._payload)
            self._update_tensors_state(TensorState.FREE)

    def _update_tensors_ptr(self) -> None:
        for tensor, tensor_info in self.tensors_info.items():
            tensor.data = self._payload[tensor_info.offset:tensor_info.end].view(tensor.shape)

    def _update_tensors_state(self, next_state: TensorState, prev_state: Optional[TensorState] = None):
        for tensor_info in self.tensors_info.values():
            if prev_state is None or tensor_info.state == prev_state:
                tensor_info.state = next_state

    def access(self) -> None:
        """
        Broadcast the chunk to synchronize the tensors across data parallel processes.
        """
        # recover the chunk on non-owner processes
        # and broadcast the chunk from the source to all processes
        if not self.is_src_rank:
            alloc_storage(self._payload)
        self.move_device(get_current_device(), update_ptr=False)
        dist.broadcast(self.data, self.global_src_rank, group=gpc.get_group(ParallelMode.DATA))

        # update tensor meta info
        self._update_tensors_ptr()
        if not self.is_src_rank:
            self._update_tensors_state(TensorState.HOLD, prev_state=TensorState.FREE)

    def move_device(self, device: torch.device, update_ptr: bool = True) -> None:
        """
        Move the chunk to a target device.

        Args:
            device (torch.device): the target device for data movement.
        """
        if self._payload.device == device:
            return
        if self._cpu_data is None:
            self.data.data = self.data.to(device)
        else:
            if device.type == 'cuda':
                # cpu -> cuda
                src = self._cpu_data
                dest = self.data
            else:
                # cuda -> cpu
                src = self.data
                dest = self._cpu_data
            alloc_storage(dest)
            dest.copy_(src)
            free_storage(src)

        if update_ptr:
            self._update_tensors_ptr()

    def reduce(self, is_all_reduce: bool = False) -> None:
        """
        Reduce or all-reduce the chunk.

        Args:
            is_all_reduce (bool): optional, whether to all-reduce the chunk. The default is false.
        """
        self.move_device(get_current_device(), update_ptr=False)
        if is_all_reduce:
            dist.all_reduce(self.data, group=gpc.get_group(ParallelMode.DATA))
        else:
            dist.reduce(self.data, self.global_src_rank, group=gpc.get_group(ParallelMode.DATA))
        self._update_tensors_ptr()
        self._update_tensors_state(TensorState.HOLD)

    def tensor_trans_state(self, tensor: torch.Tensor, tensor_state: TensorState) -> None:
        """
        Make a transition of the tensor into the next state.

        Args:
            tensor (torch.Tensor): a torch Tensor object.
            tensor_state (TensorState): the target state for transition.
        """
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
        """
        Copy data slice to the memory space indexed by the input tensor in the chunk.

        Args:
            tensor (torch.Tensor): the tensor used to retrive meta information
            data_slice (torch.Tensor): the tensor to be copied to the chunk
        """
        tensor_info = self.tensors_info[tensor]
        self._payload[tensor_info.offset:tensor_info.end].copy_(data_slice.view(-1))
        tensor.data = self._payload[tensor_info.offset:tensor_info.end].view(tensor.shape)

    @property
    def can_release(self) -> bool:
        """
        Check whether the chunk can be released.
        """
        for tensor_info in self.tensors_info.values():
            if tensor_info.state != TensorState.HOLD:
                return False
        return True

    @property
    def can_move_device(self) -> bool:
        """
        Check whether the chunk can be moved across devices.
        """
        for tensor_info in self.tensors_info.values():
            if tensor_info.state in (TensorState.COMPUTE, TensorState.READY_FOR_REDUCE):
                return False
        return True

    @property
    def can_reduce(self) -> bool:
        """
        Check whether the chunk can be reduced.
        """
        for tensor_info in self.tensors_info.values():
            if tensor_info.state != TensorState.READY_FOR_REDUCE:
                return False
        return True

    @property
    def is_empty(self) -> bool:
        """
        Check whether the chunk is empty.
        """
        return is_storage_empty(self._payload)

    def __repr__(self) -> str:
        return f'Chunk: src rank={self.src_rank} ,size={self.size}, utilization={self.utilized_size/self.size*100:.2f}%, freed={self.is_empty}, tensor states={[info.state.name for info in self.tensors_info.values()]}'

    @property
    def has_inf_or_nan(self) -> bool:
        """
        Check if the chunk has inf or nan values.
        """
        return torch.isinf(self._payload[:self.utilized_size]).any().item() or \
            torch.isnan(self._payload[:self.utilized_size]).any().item()

    def copy_(self, dest_chunk: 'Chunk'):
        """
        Copy the data of this chunk to a destination chunk.
        """
        assert not self.is_empty
        assert not dest_chunk.is_empty
        assert self.size == dest_chunk.size
        assert self.utilized_size == dest_chunk.utilized_size
        self._payload.copy_(dest_chunk._payload)
        self._update_tensors_ptr()

    @property
    def device_type(self) -> str:
        """
        Get the device type of the chunk.
        """
        return self._payload.device.type

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, __o: object) -> bool:
        return self is __o

    def get_tensors(self) -> List[torch.Tensor]:
        return list(self.tensors_info.keys())

    @property
    def _payload(self) -> torch.Tensor:
        if self._cpu_data is None or is_storage_empty(self._cpu_data):
            return self.data
        return self._cpu_data


class ChunkManager:
    """
    A manager class to manipulate the tensors in chunks.

    Args:
        chunk_size (int): the size of a chunk.
        enable_distributed_storage (bool): optional, allow for distributed storage of a chunk. The default is false.
        init_device (torch.device): optional, the device on which the chunk is initialized. The default is None.
    """

    def __init__(self,
                 chunk_size: Optional[int],
                 enable_distributed_storage: bool = False,
                 init_device: Optional[torch.device] = None) -> None:
        assert chunk_size is None or chunk_size > 0
        self.chunk_size = chunk_size
        self.enable_distributed_storage = enable_distributed_storage
        self.device = init_device or get_current_device()
        self.chunk_groups: Dict[str, Deque[Chunk]] = {}
        self.groups_force_data_on_cuda: Dict[str, bool] = {}
        self.tensor_chunk_map: Dict[torch.Tensor, Chunk] = {}
        self.accessed_chunks: Set[Chunk] = set()
        self.lazy_release_tensors: List[torch.Tensor] = []
        if enable_distributed_storage and chunk_size is None:
            self.rank_load: Dict[str, torch.Tensor] = {}
        self.total_mem: Dict[str, int] = {'cpu': 0, 'cuda': 0}

    def create_group(self, group_name: str, force_data_on_cuda: bool = False) -> None:
        """Create a chunk group.

        Args:
            group_name (str): group name
            force_data_on_cuda (bool, optional): If True, the data of chunks in this group is always on cuda.. Defaults to False.
        """
        assert group_name not in self.chunk_groups
        self.chunk_groups[group_name] = deque()
        self.groups_force_data_on_cuda[group_name] = force_data_on_cuda

    def append_tensor(self, tensor: torch.Tensor, group_name: str) -> None:
        """
        Append a tensor to a chunk.

        Args:
            tensor (torch.Tensor): a tensor to append to the chunk.
            group_name (str): the name of the chunk group.
        """
        assert tensor not in self.tensor_chunk_map
        if self.chunk_size is not None and tensor.numel() > self.chunk_size:
            raise ValueError(
                f'Cannot create chunk, got tensor numel ({tensor.numel()}) > chunk size ({self.chunk_size})')
        try:
            # append the tensor to the last chunk
            self.chunk_groups[group_name][-1].append(tensor)
        except (IndexError, ChunkFullError):
            # the except statement will be triggered when there is no chunk or
            # the last chunk in the chunk group is full
            # this will create a new chunk and allocate this chunk to its corresponding process
            chunk_size = self.chunk_size or tensor.numel()
            src_rank = self._get_next_src_rank(group_name)
            chunk = Chunk(chunk_size,
                          src_rank,
                          tensor.dtype,
                          self.device,
                          force_data_on_cuda=self.groups_force_data_on_cuda[group_name])

            if self.enable_distributed_storage and self.chunk_size is None:
                self.rank_load[group_name][src_rank] += chunk_size

            self.chunk_groups[group_name].append(chunk)
            chunk.append(tensor)
            if not chunk.is_empty:
                self.total_mem[chunk.device_type] += chunk.mem
        self.tensor_chunk_map[tensor] = self.chunk_groups[group_name][-1]
        if not self.enable_distributed_storage:
            # as distributed storage is not enabled, there is no need to broadcast
            # chunks, thus we set these chunks as accessed
            self.accessed_chunks.add(self.chunk_groups[group_name][-1])

    def _get_next_src_rank(self, group_name: str) -> int:
        if not self.enable_distributed_storage:
            # the chunk is owned by the current rank if no distributed storage is enabled
            return gpc.get_local_rank(ParallelMode.DATA)
        if self.chunk_size is None:
            if group_name not in self.rank_load:
                self.rank_load[group_name] = torch.zeros(gpc.get_world_size(ParallelMode.DATA), dtype=torch.int64)

            # the process owning the tensor will be the process with the smallest number of elements
            src_rank = torch.argmin(self.rank_load[group_name]).item()
        else:
            # chunk is owned by processes in a round-robin fashion
            chunk_idx = len(self.chunk_groups[group_name])
            src_rank = chunk_idx % gpc.get_world_size(ParallelMode.DATA)
        return src_rank

    def access_chunk(self, chunk: Chunk) -> None:
        """
        Synchronize the chunks via broadcast.

        Args:
            chunk (Chunk): the chunk to synchronize.
        """
        if chunk in self.accessed_chunks:
            if chunk.device_type != 'cuda':
                self.total_mem[chunk.device_type] -= chunk.mem
                chunk.move_device(get_current_device())
                self.total_mem[chunk.device_type] += chunk.mem
            return
        if not chunk.is_empty:
            # as tensor is moved to the target device
            # the memory consumption of the original device is reduced
            self.total_mem[chunk.device_type] -= chunk.mem
        chunk.access()
        self.accessed_chunks.add(chunk)
        self.total_mem[chunk.device_type] += chunk.mem

    def release_chunk(self, chunk: Chunk) -> None:
        """
        Release the memory space of a chunk.

        Args:
            chunk (Chunk): the chunk to release memory space
        """

        if not self.enable_distributed_storage:
            return
        if chunk not in self.accessed_chunks:
            return
        if chunk.can_release:
            chunk.release()
            self.accessed_chunks.remove(chunk)
            if chunk.is_empty:
                # update the memory consumption after releasing
                self.total_mem[chunk.device_type] -= chunk.mem

    def move_chunk(self, chunk: Chunk, device: torch.device, update_ptr: bool = True) -> None:
        """
        Move the chunk to the target device.

        Args:
            chunk (Chunk): the chunk to move to target device
            device (torch.device): target device
        """
        if chunk.data.device == device:
            return
        if chunk.can_move_device and not chunk.is_empty:
            self.total_mem[chunk.device_type] -= chunk.mem
            chunk.move_device(device, update_ptr=update_ptr)
            self.total_mem[chunk.device_type] += chunk.mem

    def trans_tensor_state(self, tensor: torch.Tensor, state: TensorState) -> None:
        """
        Transit tensor state according to pre-defined state machine.

        Args:
            tensor (torch.Tensor): the tensor for state transititon
            state (TensorState): next tensor state for transtition
        """
        chunk = self.tensor_chunk_map[tensor]
        chunk.tensor_trans_state(tensor, state)

    def reduce_chunk(self, chunk: Chunk) -> bool:
        """
        Reduce or all reduce the chunk. If enable_distributed_storage is true, all-reduce is used.
        Otherwise, this method uses reduce.

        Args:
            chunk (Chunk): the chunk for reduction.
        """
        if not chunk.can_reduce:
            return False
        self.total_mem[chunk.device_type] -= chunk.mem
        chunk.reduce(is_all_reduce=not self.enable_distributed_storage)
        self.total_mem[chunk.device_type] += chunk.mem
        return True

    def copy_tensor_to_chunk_slice(self, tensor: torch.Tensor, data: torch.Tensor) -> None:
        """
        Copy data to the chunk.

        Args:
            tensor (torch.Tensor): the tensor used to retrive meta information
            data (torch.Tensor): the tensor to be copied to the chunk
        """
        chunk = self.tensor_chunk_map[tensor]
        chunk.copy_tensor_to_chunk_slice(tensor, data)

    def get_chunk(self, tensor: torch.Tensor) -> Chunk:
        """
        Return the chunk owning the tensor.

        Args:
            tensor (torch.Tensor): a torch tensor object
        """
        return self.tensor_chunk_map[tensor]

    def add_lazy_release_tensors(self, tensors: List[torch.Tensor]) -> None:
        """
        Add tensors to the buffer for lazy release.

        Args:
            tensors (List[torch.Tensor]): the tensors to be released lazily
        """
        self.lazy_release_tensors.extend(tensors)

    def exec_lazy_release(self) -> None:
        """
        Execute release for tensors added to the lazy release buffer.
        """

        for chunk in self.get_chunks(self.lazy_release_tensors):
            self.release_chunk(chunk)
        self.lazy_release_tensors.clear()

    def __repr__(self) -> str:
        msg = f'Rank {gpc.get_local_rank(ParallelMode.DATA)}:\n'
        msg += 'Total memory: ' + ', '.join([f'{k}={v}B' for k, v in self.total_mem.items()]) + '\n'
        for group_name, group in self.chunk_groups.items():
            msg += f'Group {group_name}:\n'
            for i, chunk in enumerate(group):
                msg += f'[{i}] {chunk}\n'
        return msg

    @staticmethod
    def get_chunk_util(chunk_size: int, params_numel: List[int]) -> float:
        """
        Calculate the utilization rate of a chunk.

        Args:
            chunk_size (int): the size of a chunk
            params_numel (List[int]): the list of integers representing the number of elements of parameters
        """
        assert len(params_numel) > 0
        total_size = 0
        total_utilized_size = 0
        cur_chunk_utilized_size = 0
        for size in params_numel:
            assert chunk_size >= size
            total_utilized_size += size
            if total_size == 0 or cur_chunk_utilized_size + size > chunk_size:
                total_size += chunk_size
                cur_chunk_utilized_size = 0
            cur_chunk_utilized_size += size
        return total_utilized_size / total_size

    @staticmethod
    def search_chunk_size(module: torch.nn.Module,
                          search_range: int,
                          n_grids: int,
                          min_chunk_size: Optional[int] = None) -> int:
        """
        Search for the chunk size for optimal chunk utilization.

        Args:
            module (torch.nn.Module): a torch module object
            search_range (int): the range of chunk size to search. The actual search range will be from
                 max(min_chunk_size, max_param_size) to max(min_chunk_size, max_param_size) + search_range.
            n_grids (int): the number of intervals in the search range
            min_chunk_size (int): optional, the minimum size for a chunk. The default is None.

        """
        assert search_range % n_grids == 0
        # TODO(ver217): sort params and filter unused ones
        params_numel = [p.numel() for p in module.parameters()]
        max_param_numel = max(params_numel)
        if min_chunk_size is not None:
            assert min_chunk_size >= max_param_numel
        else:
            min_chunk_size = max_param_numel
        step_size = search_range // n_grids
        max_chunk_util = -1
        best_chunk_size = -1
        for chunk_size in range(min_chunk_size, min_chunk_size + search_range + 1, step_size):
            chunk_util = ChunkManager.get_chunk_util(chunk_size, params_numel)
            if chunk_util > max_chunk_util:
                max_chunk_util = chunk_util
                best_chunk_size = chunk_size
        return best_chunk_size

    def copy_chunk_group(self, dest_group_name: str, src_group_name: str):
        """
        Copy chunk data from one group to another group.

        Args:
            dest_group_name (str): the destination group which receives the copied data
            src_group_name (str): the source group which provides the data to copy
        """
        for dest_chunk, src_chunk in zip(self.chunk_groups[dest_group_name], self.chunk_groups[src_group_name]):
            if not dest_chunk.is_empty:
                dest_chunk.copy_(src_chunk)

    def get_chunks(self, tensors: Iterable[torch.Tensor]) -> Tuple[Chunk, ...]:
        """
        Get all chunks owning the input tensors.

        Args:
            tensors (Iterable[torch.Tensor]): the tensors used to look for chunks
        """
        chunks = []
        for tensor in tensors:
            chunk = self.get_chunk(tensor)
            if chunk not in chunks:
                chunks.append(chunk)
        return tuple(chunks)

    def add_extern_static_tensor(self, tensor: torch.Tensor) -> None:
        """Add extern static tensor to chunk manager. 
        Those tensors won't be managed by chunk manager, but we want to monitor memory usage of them.
        They are "static", which means their shape, dtype, device never change.
        Thus, their memory usage never changes.

        Args:
            tensor (torch.Tensor): An extern static tensor. E.g. optimizer state.
        """
        assert tensor not in self.tensor_chunk_map
        self.total_mem[tensor.device.type] += tensor.numel() * tensor.element_size()
