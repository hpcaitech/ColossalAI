from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from colossalai.tensor import ProcessGroup as ColoProcessGroup
from colossalai.utils import get_current_device


class TensorState(Enum):
    FREE = 0
    COMPUTE = 1
    HOLD = 2
    HOLD_AFTER_BWD = 3
    READY_FOR_REDUCE = 4


STATE_TRANS = ((TensorState.FREE, TensorState.HOLD), (TensorState.FREE, TensorState.COMPUTE),
               (TensorState.HOLD, TensorState.FREE), (TensorState.HOLD, TensorState.COMPUTE), (TensorState.COMPUTE,
                                                                                               TensorState.HOLD),
               (TensorState.COMPUTE, TensorState.HOLD_AFTER_BWD), (TensorState.HOLD_AFTER_BWD, TensorState.COMPUTE),
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
    _total_number = 0

    def __init__(self,
                 chunk_size: int,
                 process_group: ColoProcessGroup,
                 dtype: torch.dtype,
                 init_device: Optional[torch.device] = None,
                 cpu_shard_init: bool = False,
                 keep_gathered: bool = False,
                 pin_memory: bool = False) -> None:
        """
        Chunk: A container owning a piece of contiguous memory space for tensors
        Here we use all-gather operation to gather the whole chunk.
        Currently, Chunk is exclusively used for DDP and ZeRO DDP and it doesn't support unused parameters.
        It is designed to make the full use of communication and PCIE bandwidth.

        Args:
            chunk_size (int): the number of elements in the chunk
            process_group (ColoProcessGroup): the process group of this chunk
            dtype (torch.dtype): the data type of the chunk
            init_device (torch.device): optional, During the chunk construction process, where the tensor is stored.
                The default value is None, which is the current GPU
            cpu_shard_init (bool): a flag indicates the local chunk shard is resident on CPU.
            keep_gathered (bool): optional, if True, this chunk is always gathered in CUDA memory
            pin_memory (bool): optional, if True, this chunk always has a shard copied in pinned CPU memory
        """
        self.count_id = Chunk._total_number
        Chunk._total_number += 1

        self.chunk_size = chunk_size
        self.utilized_size = 0

        self.torch_pg = process_group.dp_process_group()
        self.pg_size = dist.get_world_size(self.torch_pg)
        self.pg_rank = dist.get_rank(self.torch_pg)

        # the chunk size should be divisible by the dp degree
        if not keep_gathered:
            assert chunk_size % self.pg_size == 0
        self.shard_size = chunk_size // self.pg_size
        self.shard_begin = self.shard_size * self.pg_rank
        self.shard_end = self.shard_begin + self.shard_size
        self.valid_end = self.shard_size

        self.dtype = dtype
        device = init_device or get_current_device()

        # chunk_temp is a global chunk, which only exists during building the chunks.
        self.chunk_temp = torch.zeros(chunk_size, dtype=dtype, device=device)    # keep all zero

        self.cuda_global_chunk = None    # we force cuda_global_chunk located in CUDA

        # cuda local chunk, which is sharded on GPUs
        self.cuda_shard = None
        # cpu local chunk, which is sharded on CPUs
        self.cpu_shard = None
        # is the chunks gathers, which means chunks are duplicated on each process,
        # and we should use the cuda_global_chunk.
        self.is_gathered = True

        # configure the init device of the shard
        # no-offload default: fp16, fp32 -> CUDA
        # offload default: fp16, fp32 -> CPU
        self.shard_device = torch.device("cpu") if cpu_shard_init else get_current_device()

        self.chunk_mem = self.chunk_size * self.chunk_temp.element_size()
        self.shard_mem = self.chunk_mem // self.pg_size

        # each tensor is associated with a TensorInfo to track its meta info
        # (state, offset, end)
        self.tensors_info: Dict[torch.Tensor, TensorInfo] = {}
        # the total number of tensors in the chunk
        self.num_tensors = 0

        # Record the number of tensors in different states
        self.tensor_state_cnter: Dict[TensorState, int] = dict()
        for state in TensorState:
            self.tensor_state_cnter[state] = 0

        # If a chunk is kept gathered,
        # they are treated the same as that of the parameters in DDP during training.
        self.keep_gathered = keep_gathered
        if self.keep_gathered:
            pin_memory = False    # since this chunk is gathered, it doesn't need to pin

        # if pin_memory is True, we allocate a piece of CPU pin-memory
        # for it all the time
        self.pin_memory = pin_memory

        # we introduce the paired chunk here
        # it refers to another chunk having the same parameters
        # but with different dtype(such as fp16_chunk.paired_chunk -> fp32_chunk
        self.paired_chunk = None
        # if this chunk is synchronized with the optimizer, the flag is True
        self.optim_sync_flag = True
        # if the cpu_shard has been visited during the training step, the flag is True
        self.cpu_vis_flag = False

        # whether to record l2 norm for the gradient clipping calculation
        self.l2_norm_flag = False
        self.l2_norm = None

    @property
    def memory_usage(self) -> Dict[str, int]:
        cuda_memory = 0
        cpu_memory = 0

        if self.chunk_temp is not None:
            # this chunk is not closed
            if self.chunk_temp.device.type == 'cuda':
                cuda_memory += self.chunk_mem
            else:
                cpu_memory += self.chunk_mem
        else:
            if self.is_gathered:
                cuda_memory += self.chunk_mem
            if self.cuda_shard is not None:
                cuda_memory += self.shard_mem
            if self.cpu_shard is not None:
                cpu_memory += self.shard_mem

        return dict(cuda=cuda_memory, cpu=cpu_memory)

    @property
    def device_type(self) -> str:
        if self.chunk_temp is not None:
            return self.chunk_temp.device.type
        else:
            if self.is_gathered:
                return 'cuda'
            elif self.cuda_shard is not None:
                return 'cuda'
            else:
                return 'cpu'

    @property
    def payload(self) -> torch.Tensor:
        # sanity check
        assert self.chunk_temp is None

        if self.is_gathered:
            return self.cuda_global_chunk
        elif self.cuda_shard is not None:
            return self.cuda_shard
        else:
            return self.cpu_shard

    @property
    def payload_mem(self) -> int:
        # sanity check
        assert self.chunk_temp is None

        if self.is_gathered:
            return self.chunk_mem
        else:
            return self.shard_mem

    @property
    def can_move(self) -> bool:
        return not self.is_gathered

    @property
    def can_release(self) -> bool:
        if self.keep_gathered:
            return False
        else:
            return self.tensor_state_cnter[TensorState.HOLD] + \
                   self.tensor_state_cnter[TensorState.HOLD_AFTER_BWD] == self.num_tensors

    @property
    def can_reduce(self):
        return self.tensor_state_cnter[TensorState.READY_FOR_REDUCE] == self.num_tensors

    @property
    def has_inf_or_nan(self) -> bool:
        """Check if the chunk has inf or nan values on CUDA.
        """
        if self.is_gathered:
            valid_tensor = self.cuda_global_chunk[:self.utilized_size]
        else:
            assert self.cuda_shard is not None    # only check on CUDA
            valid_tensor = self.cuda_shard[:self.valid_end]

        return torch.isinf(valid_tensor).any().item() | torch.isnan(valid_tensor).any().item()

    def set_l2_norm(self) -> None:
        """Record l2 norm of this chunks on CUDA.
        """
        assert self.l2_norm is None, "you are calculating the l2 norm twice"
        if self.is_gathered:
            valid_tensor = self.cuda_global_chunk[:self.utilized_size]
        else:
            assert self.cuda_shard is not None    # calculate on CUDA
            valid_tensor = self.cuda_shard[:self.valid_end]
        chunk_l2_norm = valid_tensor.data.float().norm(2)
        self.l2_norm = chunk_l2_norm.item()**2

    def append_tensor(self, tensor: torch.Tensor):
        """Add a tensor to the chunk.

        Args:
            tensor (torch.Tensor): a tensor to be added to the chunk
        """
        # sanity check
        assert self.chunk_temp is not None
        assert tensor.dtype == self.dtype

        new_utilized_size = self.utilized_size + tensor.numel()
        # raise exception when the chunk size is exceeded
        if new_utilized_size > self.chunk_size:
            raise ChunkFullError

        self.chunk_temp[self.utilized_size:new_utilized_size].copy_(tensor.data.flatten())
        assert type(self.chunk_temp) == torch.Tensor, "copy_tensor_to_chunk_slice must use a torch tensor"
        tensor.data = self.chunk_temp[self.utilized_size:new_utilized_size].view(tensor.shape)

        # record all the information about the tensor
        self.num_tensors += 1
        tensor_state = TensorState.HOLD
        self.tensors_info[tensor] = TensorInfo(tensor_state, self.utilized_size, new_utilized_size)
        self.tensor_state_cnter[tensor_state] += 1
        self.utilized_size = new_utilized_size

    def close_chunk(self):
        """Close the chunk. Any tensor can't be appended to a closed chunk later.
        """
        # sanity check
        assert self.chunk_temp is not None

        # calculate the valid end for each shard
        if self.utilized_size <= self.shard_begin:
            self.valid_end = 0
        elif self.utilized_size < self.shard_end:
            self.valid_end = self.utilized_size - self.shard_begin

        if self.chunk_temp.device.type == 'cpu':
            self.cuda_global_chunk = self.chunk_temp.to(get_current_device())
            self.__update_tensors_ptr()
        else:
            self.cuda_global_chunk = self.chunk_temp
        self.chunk_temp = None

        self.__scatter()
        # gathered chunk never have shard attribute
        if self.keep_gathered:
            return

        if self.pin_memory or self.shard_device.type == 'cpu':
            self.cpu_shard = torch.empty(self.shard_size, dtype=self.dtype, pin_memory=self.pin_memory)
            self.cpu_shard.copy_(self.cuda_shard)
            self.cpu_vis_flag = True    # cpu_shard has been visited

        if self.shard_device.type == 'cpu':
            self.cuda_shard = None

    def shard_move(self, device: torch.device, force_copy: bool = False):
        """Move the shard tensor in the chunk.

        Args:
            device: the device to which the shard will move
            force_copy: if True, copy function is called mandatorily
        """
        # sanity check
        assert not self.is_gathered
        # when the current chunk is not synchronized with the optimizer
        # just use another way for the movement
        if not self.optim_sync_flag:
            assert device.type == 'cuda', "each chunk should first be moved to CUDA"
            self.__paired_shard_move()
            self.optim_sync_flag = True
            return

        if device.type == 'cuda':
            assert device == get_current_device(), "can't move chunk to another device"

            if self.cuda_shard:
                return

            self.cuda_shard = self.cpu_shard.to(get_current_device())

            if not self.pin_memory:
                self.cpu_shard = None
        elif device.type == 'cpu':
            if self.cuda_shard is None:
                return

            if self.pin_memory:
                if force_copy or not self.cpu_vis_flag:
                    self.cpu_shard.copy_(self.cuda_shard)
                # if cpu_shard has been visited
                # copy operation is not need
            else:
                self.cpu_shard = self.cuda_shard.cpu()
            self.cpu_vis_flag = True
            self.cuda_shard = None
        else:
            raise NotImplementedError

    def access_chunk(self):
        """Make the chunk usable for the parameters inside it. It's an operation done in CUDA.
        """
        # sanity check
        assert self.chunk_temp is None

        if not self.is_gathered:
            self.__gather()
        self.__update_tensors_ptr()

    def release_chunk(self):
        """Release the usable chunk. It's an operation done in CUDA.
        """
        # sanity check
        assert self.chunk_temp is None

        if self.is_gathered:
            self.__scatter()

    def reduce(self):
        """Reduce scatter all the gradients. It's an operation done in CUDA.
        """
        # sanity check
        assert self.is_gathered

        if self.pg_size == 1:
            # tricky code here
            # just move cuda_global_chunk to cuda_shard
            # the communication is not necessary
            self.__scatter()
        elif self.keep_gathered:
            # we use all-reduce here
            dist.all_reduce(self.cuda_global_chunk, group=self.torch_pg)
        else:
            self.cuda_shard = torch.empty(self.shard_size, dtype=self.dtype, device=get_current_device())

            input_list = list(torch.chunk(self.cuda_global_chunk, chunks=self.pg_size, dim=0))
            dist.reduce_scatter(self.cuda_shard, input_list, group=self.torch_pg)

            free_storage(self.cuda_global_chunk)
            self.is_gathered = False
        self.__update_tensors_state(TensorState.HOLD)

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
            return
        self.__update_one_tensor_info(self.tensors_info[tensor], tensor_state)

    def copy_tensor_to_chunk_slice(self, tensor: torch.Tensor, data_slice: torch.Tensor) -> None:
        """
        Copy data slice to the memory space indexed by the input tensor in the chunk.

        Args:
            tensor (torch.Tensor): the tensor used to retrive meta information
            data_slice (torch.Tensor): the tensor to be copied to the chunk
        """
        # sanity check
        assert self.is_gathered

        tensor_info = self.tensors_info[tensor]
        self.cuda_global_chunk[tensor_info.offset:tensor_info.end].copy_(data_slice.data.flatten())
        tensor.data = self.cuda_global_chunk[tensor_info.offset:tensor_info.end].view(tensor.shape)

    def get_valid_length(self) -> int:
        """Get the valid length of the chunk's payload.
        """
        if self.keep_gathered:
            return self.utilized_size
        else:
            return self.valid_end

    def init_pair(self, friend_chunk: 'Chunk') -> None:
        """Initialize the paired chunk.
        """
        if self.paired_chunk is None and friend_chunk.paired_chunk is None:
            self.paired_chunk = friend_chunk
            friend_chunk.paired_chunk = self
        else:
            assert self.paired_chunk is friend_chunk
            assert friend_chunk.paired_chunk is self

    def optim_update(self) -> None:
        """Update the fp16 chunks via their fp32 chunks. It's used by the optimizer.
        """
        # sanity check
        assert self.paired_chunk is not None

        friend_chunk = self.paired_chunk
        if self.is_gathered is True:
            assert friend_chunk.is_gathered is True
            self.cuda_global_chunk.copy_(friend_chunk.cuda_global_chunk)
            self.optim_sync_flag = True
        elif friend_chunk.device_type == 'cuda' and self.device_type == 'cuda':
            self.cuda_shard.copy_(friend_chunk.cuda_shard)
            self.optim_sync_flag = True
            self.cpu_vis_flag = False
        else:
            # optim_sync_flag is set to False
            # see shard_move function for more details
            assert friend_chunk.device_type == 'cpu'
            assert self.device_type == 'cpu'
            self.optim_sync_flag = False
            self.cpu_vis_flag = False

    def get_tensors(self) -> List[torch.Tensor]:
        return list(self.tensors_info.keys())

    def __gather(self):
        if not self.is_gathered:
            # sanity check
            assert self.cuda_shard is not None

            alloc_storage(self.cuda_global_chunk)
            gather_list = list(torch.chunk(input=self.cuda_global_chunk, chunks=self.pg_size, dim=0))
            dist.all_gather(gather_list, self.cuda_shard, self.torch_pg)

            self.cuda_shard = None
            self.is_gathered = True

    def __scatter(self):
        if self.keep_gathered:
            return

        if self.is_gathered:
            # sanity check
            assert self.cuda_shard is None

            self.cuda_shard = torch.empty(self.shard_size, dtype=self.dtype, device=self.cuda_global_chunk.device)

            self.cuda_shard.copy_(self.cuda_global_chunk[self.shard_begin:self.shard_end])

            free_storage(self.cuda_global_chunk)
            self.is_gathered = False

    def __paired_shard_move(self):
        assert self.paired_chunk is not None, "chunks should be paired before training"
        optim_chunk = self.paired_chunk
        assert self.chunk_size == optim_chunk.chunk_size

        # only be called when optimizer state is in CPU memory
        # the grad and param should be in the same device
        assert self.cuda_shard is None
        temp = optim_chunk.cpu_shard.to(get_current_device())
        # avoid to transform FP32 in CPU
        self.cuda_shard = temp.to(self.dtype)

        if not self.pin_memory:
            self.cpu_shard = None

    def __update_tensors_ptr(self) -> None:
        # sanity check
        assert self.is_gathered
        assert type(self.cuda_global_chunk) == torch.Tensor

        for tensor, tensor_info in self.tensors_info.items():
            tensor.data = self.cuda_global_chunk[tensor_info.offset:tensor_info.end].view(tensor.shape)

    def __update_one_tensor_info(self, tensor_info: TensorInfo, next_state: TensorState):
        self.tensor_state_cnter[tensor_info.state] -= 1
        tensor_info.state = next_state
        self.tensor_state_cnter[tensor_info.state] += 1

    def __update_tensors_state(self, next_state: TensorState, prev_state: Optional[TensorState] = None):
        for tensor_info in self.tensors_info.values():
            if prev_state is None or tensor_info.state == prev_state:
                self.__update_one_tensor_info(tensor_info, next_state)

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, __o: object) -> bool:
        return self is __o

    def __repr__(self, detailed: bool = True):
        output = [
            "Chunk Information:\n",
            "\tchunk size: {}, chunk dtype: {}, process group size: {}\n".format(self.chunk_size, self.dtype,
                                                                                 self.pg_size),
            "\t# of tensors: {}, utilized size: {}, utilized percentage: {:.2f}\n".format(
                self.num_tensors, self.utilized_size, self.utilized_size / self.chunk_size)
        ]

        def print_tensor(tensor, prefix=''):
            output.append("{}shape: {}, dtype: {}, device: {}\n".format(prefix, tensor.shape, tensor.dtype,
                                                                        tensor.device))

        if self.chunk_temp is not None:
            output.append("\tchunk temp:\n")
            print_tensor(tensor=self.chunk_temp, prefix='\t\t')

        if self.cuda_global_chunk is not None and self.cuda_global_chunk.storage().size() > 0:
            output.append("\tchunk total:\n")
            print_tensor(tensor=self.cuda_global_chunk, prefix='\t\t')

        if self.cuda_shard is not None:
            output.append("\tcuda shard:\n")
            print_tensor(tensor=self.cuda_shard, prefix='\t\t')

        if self.cpu_shard is not None:
            output.append("\tcpu shard:\n")
            print_tensor(tensor=self.cpu_shard, prefix='\t\t')

        memory_info = self.memory_usage
        output.append("\tmemory usage: cuda {}, cpu {}\n".format(memory_info['cuda'], memory_info['cpu']))

        if detailed:
            output.append("\ttensor state monitor:\n")
            for st in TensorState:
                output.append("\t\t# of {}: {}\n".format(st, self.tensor_state_cnter[st]))

        return ''.join(output)
