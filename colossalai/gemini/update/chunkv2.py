import torch
import torch.distributed as dist
from typing import Optional, Dict, List

from colossalai.utils import get_current_device
from colossalai.tensor import ProcessGroup as ColoProcessGroup
from colossalai.gemini.chunk import TensorState, STATE_TRANS, TensorInfo, ChunkFullError, \
    free_storage, alloc_storage


class ChunkV2:

    def __init__(self,
                 chunk_size: int,
                 process_group: ColoProcessGroup,
                 dtype: torch.dtype,
                 init_device: Optional[torch.device] = None,
                 keep_gathered: bool = False,
                 pin_memory: bool = False) -> None:
        """
        Chunk: A container owning a piece of contiguous memory space for tensors
        AgChunk is a kind of chunk, which uses all-gather operation to gather the whole chunk.
        This kind of chunk is exclusively used for DDP and ZeRO DDP.
        It is designed to make the full use of communication and PCIE bandwidth.

        Args:
            chunk_size (int): the number of elements in a chunk
            process_group (ColoProcessGroup): the process group of this chunk
            dtype (torch.dtype): the data type of the chunk
            init_device (torch.device): optional, the device where the tensor is initialized
                The default value is None, which is the current GPU
            keep_gathered (bool): optional, if True, this chunk is always gathered in CUDA memory
            pin_memory (bool): optional, if True, this chunk always has a shard copy in pinned CPU memory
        """

        self.chunk_size = chunk_size
        self.utilized_size = 0
        # Here, we use torch process group,
        # since ColoProcessGroup might get deprecated soon
        self.torch_pg = process_group.dp_process_group()
        self.pg_size = dist.get_world_size(self.torch_pg)
        self.pg_rank = dist.get_rank(self.torch_pg)

        # the chunk size should be able to be divied by the size of GPU
        assert chunk_size % self.pg_size == 0
        self.shard_size = chunk_size // self.pg_size
        self.shard_begin = self.shard_size * self.pg_rank
        self.shard_end = self.shard_begin + self.shard_size
        self.valid_end = self.shard_size

        self.dtype = dtype
        device = init_device or get_current_device()
        self.chunk_temp = torch.zeros(chunk_size, dtype=dtype, device=device)    # keep all zero
        self.chunk_total = None    # we force chunk_total located in CUDA
        self.cuda_shard = None    # using two attributes for the better interpretation
        self.cpu_shard = None
        self.is_gathered = True

        self.chunk_mem = self.chunk_size * self.chunk_temp.element_size()
        self.shard_mem = self.chunk_mem // self.pg_size

        # each tensor is associated with a TensorInfo to track meta info
        self.tensors_info: Dict[torch.Tensor, TensorInfo] = {}
        # the total number of all tensors
        self.num_tensors = 0
        # monitor the states of all tensors
        self.tensors_state_monitor: Dict[TensorState, int] = dict()
        for state in TensorState:
            self.tensors_state_monitor[state] = 0

        # some chunks can keep gathered all the time
        # so their computation patterns are the same as that of the parameters in DDP
        self.keep_gathered = keep_gathered
        if self.keep_gathered:
            pin_memory = False    # since this chunk is gathered, it doesn't need to pin

        # if pin_memory is True, we allocate a piece of CPU pin-memory
        # for it all the time
        self.pin_memory = pin_memory

        # we introduce the paired chunk here
        # it refers to another chunk having the same parameters
        # but with different dtype(such as fp16_chunk.mapping_chunk -> fp32_chunk
        self.paired_chunk = None
        # if the the gradient of this chunk is reduced, the flag is True
        # so the flag is False for unused parameters
        self.grad_reduced_flag = False
        # if this chunk is synchronized with the optimizer, the flag is True
        self.optim_sync_flag = True
        # if the cpu_shard has been visited during the training step, the flag is True
        self.cpu_vis_flag = False

    @property
    def memory_usage(self):
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
    def device_type(self):
        if self.chunk_temp is not None:
            return self.chunk_temp.device.type
        else:
            if self.is_gathered:
                return 'cuda'
            elif self.cuda_shard is not None:
                return 'cuda'
            else:
                return 'cpu'

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
        self.tensors_state_monitor[tensor_state] += 1
        self.utilized_size = new_utilized_size

    def close_chunk(self, shard_dev: Optional[torch.device] = None):
        """Close the chunk. Any tensor can't be appended to a closed chunk.
        """
        # sanity check
        assert self.chunk_temp is not None

        # calculate the valid end for each shard
        if self.utilized_size <= self.shard_begin:
            self.valid_end = 0
        elif self.utilized_size < self.shard_end:
            self.valid_end = self.utilized_size - self.shard_begin

        if self.chunk_temp.device.type == 'cpu':
            self.chunk_total = self.chunk_temp.to(get_current_device())
        else:
            self.chunk_total = self.chunk_temp
        self.chunk_temp = None

        self.__scatter()

        if self.keep_gathered:
            if shard_dev is None:
                shard_dev = get_current_device()
            else:
                assert shard_dev.type == 'cuda'
        elif shard_dev is None:
            shard_dev = torch.device('cpu')

        if self.pin_memory or shard_dev.type == 'cpu':
            self.cpu_shard = torch.empty(self.shard_size, dtype=self.dtype, pin_memory=self.pin_memory)
            self.cpu_shard.copy_(self.cuda_shard)
            self.cpu_vis_flag = True    # cpu_shard has been visited

        if shard_dev.type == 'cpu':
            self.cuda_shard = None

    def shard_move(self, device: torch.device, force_copy: bool = False):
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
        """Make the chunk usable for the parameters inside it.
        It is an operation done in CUDA.
        """
        # sanity check
        assert self.chunk_temp is None

        if not self.is_gathered:
            self.__gather()
        self.__update_tensors_ptr()

    def release_chunk(self):
        """Release the usable chunk.
        It is an operation done in CUDA.
        """
        # sanity check
        assert self.chunk_temp is None

        if self.is_gathered:
            self.__scatter()

    def reduce(self):
        """Reduce scatter all the gradients.
        It is an operation done in CUDA.
        """
        # sanity check
        assert self.is_gathered

        if self.pg_size == 1:
            # tricky code here
            # just move chunk_total to cuda_shard
            # the communication is not necessary
            self.__scatter()
        elif self.keep_gathered:
            # we use all-reduce here
            dist.all_reduce(self.chunk_total, group=self.torch_pg)
        else:
            self.cuda_shard = torch.empty(self.shard_size, dtype=self.dtype, device=get_current_device())

            input_list = list(torch.chunk(self.chunk_total, chunks=self.pg_size, dim=0))
            dist.reduce_scatter(self.cuda_shard, input_list, group=self.torch_pg)

            free_storage(self.chunk_total)
            self.is_gathered = False
        self.__update_tensors_state(TensorState.HOLD)
        self.grad_reduced_flag = True

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
        self.chunk_total[tensor_info.offset:tensor_info.end].copy_(data_slice.data.flatten())
        tensor.data = self.chunk_total[tensor_info.offset:tensor_info.end].view(tensor.shape)

    @property
    def can_move(self) -> bool:
        return not self.is_gathered

    @property
    def can_release(self) -> bool:
        if self.keep_gathered:
            return False
        else:
            return self.tensors_state_monitor[TensorState.HOLD] + \
                   self.tensors_state_monitor[TensorState.HOLD_AFTER_BWD] == self.num_tensors

    @property
    def can_reduce(self):
        return self.tensors_state_monitor[TensorState.READY_FOR_REDUCE] == self.num_tensors

    @property
    def has_inf_or_nan(self) -> bool:
        """
        Check if the chunk has inf or nan values in CUDA.
        """
        if self.is_gathered:
            valid_tensor = self.chunk_total[:self.utilized_size]
        else:
            assert self.cuda_shard is not None    # only check in CUDA
            valid_tensor = self.cuda_shard[:self.valid_end]

        return torch.isinf(valid_tensor).any().item() | torch.isnan(valid_tensor).any().item()

    def __gather(self):
        if not self.is_gathered:
            # sanity check
            assert self.cuda_shard is not None

            if self.pg_size == 1:
                self.chunk_total = self.cuda_shard
            else:
                alloc_storage(self.chunk_total)
                gather_list = list(torch.chunk(input=self.chunk_total, chunks=self.pg_size, dim=0))
                dist.all_gather(gather_list, self.cuda_shard, self.torch_pg)

            self.cuda_shard = None
            self.is_gathered = True

    def __scatter(self):
        if self.keep_gathered:
            return

        if self.is_gathered:
            # sanity check
            assert self.cuda_shard is None

            self.cuda_shard = torch.empty(self.shard_size, dtype=self.dtype, device=self.chunk_total.device)

            self.cuda_shard.copy_(self.chunk_total[self.shard_begin:self.shard_end])

            free_storage(self.chunk_total)
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
        assert type(self.chunk_total) == torch.Tensor

        for tensor, tensor_info in self.tensors_info.items():
            tensor.data = self.chunk_total[tensor_info.offset:tensor_info.end].view(tensor.shape)

    def __update_one_tensor_info(self, tensor_info: TensorInfo, next_state: TensorState):
        self.tensors_state_monitor[tensor_info.state] -= 1
        tensor_info.state = next_state
        self.tensors_state_monitor[tensor_info.state] += 1

    def __update_tensors_state(self, next_state: TensorState, prev_state: Optional[TensorState] = None):
        for tensor_info in self.tensors_info.values():
            if prev_state is None or tensor_info.state == prev_state:
                self.__update_one_tensor_info(tensor_info, next_state)

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, __o: object) -> bool:
        return self is __o

    def __repr__(self, detailed: bool = False):
        output = [
            "AgChunk Information:\n",
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

        if self.chunk_total is not None and self.chunk_total.storage().size() > 0:
            output.append("\tchunk total:\n")
            print_tensor(tensor=self.chunk_total, prefix='\t\t')

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
                output.append("\t\t# of {}: {}\n".format(st, self.tensors_state_monitor[st]))

        return ''.join(output)

    def get_tensors(self) -> List[torch.Tensor]:
        return list(self.tensors_info.keys())
