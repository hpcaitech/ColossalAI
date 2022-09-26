import torch
import torch.distributed as dist
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List

from colossalai.utils import get_current_device
from colossalai.tensor import ProcessGroup as ColoProcessGroup


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
                 process_group: ColoProcessGroup,
                 dtype: torch.dtype,
                 init_device: Optional[torch.device] = None,
                 force_data_on_cuda: bool = False) -> None:
        self.size = chunk_size
        self.utilized_size = 0
        self.src_rank = src_rank
        self.process_group = process_group
        self.is_src_rank = process_group.dp_local_rank() == src_rank
        self.global_src_rank = process_group.get_ranks_in_dp()[src_rank]
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
            self._payload[self.utilized_size:new_utilized_size].copy_(tensor.flatten())
            tensor_state = TensorState.HOLD
            assert type(self._payload) == torch.Tensor, "copy_tensor_to_chunk_slice must use a torch tensor"
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
        assert type(self._payload) == torch.Tensor
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
        dist.broadcast(self.data, self.global_src_rank, group=self.process_group.dp_process_group())

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
            dist.all_reduce(self.data, group=self.process_group.dp_process_group())
        else:
            dist.reduce(self.data, self.global_src_rank, group=self.process_group.dp_process_group())
        self._update_tensors_ptr()
        self._update_tensors_state(TensorState.HOLD)

    def tensor_trans_state(self, tensor: torch.Tensor, tensor_state: TensorState) -> None:
        """
        Make a transition of the tensor into the next state.

        Args:
            tensor (torch.Tensor): a torch Tensor object.
            tensor_state (TensorState): the target state for transition.
        """

        # As the gradient hook can be triggered either before or after post-backward
        # tensor's state can be compute -> hold_after_bwd -> ready_for_reduce
        # or compute -> ready_for_reduce -> hold_after_bwd
        # the second one is invalid, we just ignore ready_for_reduce -> hold_after_bwd
        # this function only apply valid state transformation
        # invalid calls will be ignored and nothing changes
        if (self.tensors_info[tensor].state, tensor_state) not in STATE_TRANS:
            # print(
            #     f'WARNING: Rank{self.process_group.rank()} apply invalid state trans: {self.tensors_info[tensor].state} to {tensor_state}'
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
        self._payload[tensor_info.offset:tensor_info.end].copy_(data_slice.flatten())
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
