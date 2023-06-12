from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.elixir.cuda import gpu_device
from colossalai.elixir.tensor import FakeTensor

from .memory_pool import MemoryPool, TensorBlock
from .states import TensorState, validate_tensor_state_update


class ChunkFullError(Exception):
    pass


@dataclass
class TensorInfo:
    state: TensorState
    fake_data: FakeTensor
    offset: int
    end: int


class Chunk:
    """Chunk is a type of data structure to store tensors.
    It allows us to store a sequence of tensors into one continuous memory block.
    Moreover, Chunk manages the storage of tensors in a distributed way.
    Normally, a chunk is scattered across its process group.
    When a tensor in this chunk should be used later, the chunk can be gathered by access_chunk.
    When the training is done, the chunk can be scattered by reduce_chunk.

    args:
        rcache: the memory pool to store replicated chunks
        chunk_size: the size of the chunk
        chunk_dtype: the dtype of the chunk
        process_group: the torch communication group of the chunk
        temp_device: the device to store the temporary chunk when initializing
        shard_device: the device to store the shard of the scattered chunk
        rcache_fused: whether this chunk is fused in rcache without eviction
        cpu_pin_memory: whether this chunk use cpu pin memory for its shard
    """
    total_count = 0

    def __init__(
        self,
        rcache: MemoryPool,
        chunk_size: int,
        chunk_dtype: torch.dtype,
        process_group: ProcessGroup,
        temp_device: Optional[torch.device] = None,
        shard_device: Optional[torch.device] = None,
        rcache_fused: bool = False,    # whether this chunk is used in ZeRO2
        cpu_pin_memory: bool = False    # whether this chunk has a permanent copy in cpu
    ) -> None:

        self.chunk_id: int = Chunk.total_count
        Chunk.total_count += 1
        # set replicated cache pool
        self.rcache: MemoryPool = rcache

        self.chunk_size: int = chunk_size
        self.chunk_dtype: torch.dtype = chunk_dtype
        self.utilized_size: int = 0

        self.torch_pg: ProcessGroup = process_group
        self.pg_size: int = dist.get_world_size(self.torch_pg)
        self.pg_rank: int = dist.get_rank(self.torch_pg)

        # the chunk size should be divisible by the dp degree
        assert chunk_size % self.pg_size == 0

        self.shard_size: int = chunk_size // self.pg_size
        self.shard_begin: int = self.shard_size * self.pg_rank
        self.shard_end: int = self.shard_begin + self.shard_size
        self.valid_end: int = self.shard_size + 1    # set to an illegal number

        # notice: release blocks reserved by Pytorch
        torch.cuda.empty_cache()
        # rcache block, the global replicated chunk in R cache
        self.rcb: Optional[TensorBlock] = None
        self.rcache_fused: bool = rcache_fused
        self._my_block = None
        self.is_replica: bool = True
        # allocate a private block for fused chunks
        if self.rcache_fused:
            self._my_block = rcache.get_private_block(chunk_size, chunk_dtype)

        temp_device: torch.device = temp_device or gpu_device()
        # chunk_temp is a global chunk, which only exists during building the chunks.
        # keep all elements to zero
        self.chunk_temp: Optional[torch.Tensor] = None
        if rcache_fused:
            self.chunk_temp = self._my_block.payload
            torch.zero_(self.chunk_temp)
        else:
            self.chunk_temp = torch.zeros(chunk_size, dtype=chunk_dtype, device=temp_device)

        # configure the init device of the shard
        # no-offload default: fp16, fp32 -> CUDA
        # offload default: fp16, fp32 -> CPU
        shard_device: torch.device = shard_device or torch.device('cpu')
        pin_flag: bool = cpu_pin_memory and shard_device.type == 'cpu'
        # chunk.shard is a local chunk
        # it is desinged to exist permanently
        self.shard: torch.Tensor = torch.empty(self.shard_size,
                                               dtype=chunk_dtype,
                                               device=shard_device,
                                               pin_memory=pin_flag)

        # calculate the memory occupation of the chunk and the shard
        self.chunk_memo: int = self.chunk_size * self.chunk_temp.element_size()
        self.shard_memo: int = self.chunk_memo // self.pg_size

        # each tensor is associated with a TensorInfo to track its meta info
        # (state, shape, offset, end)
        self.tensors_info: Dict[torch.Tensor, TensorInfo] = {}
        # the total number of tensors in the chunk
        self.num_tensors: int = 0

        # Record the number of tensors in different states
        self.tensor_state_cnter: Dict[TensorState, int] = dict()
        for state in TensorState:
            self.tensor_state_cnter[state] = 0

        # we introduce the paired chunk here
        # it refers to another chunk having the same parameters
        # but with different dtype(such as fp16_chunk.paired_chunk -> fp32_chunk
        self.paired_chunk = None
        # if this chunk is synchronized with the optimizer, the flag is True
        self.optim_sync_flag = True

        # whether to record l2 norm for the gradient clipping calculation
        self.l2_norm_flag = False
        self.l2_norm = None
        # whether it overflows after the reduction
        self.overflow = False

    @property
    def prepared_block(self):
        return self._my_block

    @property
    def is_init(self):
        return self.chunk_temp is not None

    @property
    def in_rcache(self):
        return self.rcb is not None

    @property
    def shard_device(self):
        return self.shard.device

    @property
    def memory_usage(self) -> Dict[str, int]:
        cuda_memory = 0
        cpu_memory = 0

        # this chunk is not closed
        if self.is_init:
            if self.chunk_temp.device.type == 'cuda':
                cuda_memory += self.chunk_memo
            else:
                cpu_memory += self.chunk_memo

        # this chunk is on the rcache
        if self.in_rcache:
            cuda_memory += self.rcb.memo_occ

        # calculate the occupation of the chunk shard
        if self.shard_device.type == 'cuda':
            cuda_memory += self.shard_memo
        elif self.shard_device.type == 'cpu':
            cpu_memory += self.shard_memo
        else:
            raise NotImplementedError

        return dict(cuda=cuda_memory, cpu=cpu_memory)

    @property
    def payload(self) -> torch.Tensor:
        if self.is_init:
            return self.chunk_temp

        if self.in_rcache:
            return self.rcb.payload
        else:
            return self.shard

    @property
    def shard_move_check(self) -> bool:
        return not self.in_rcache

    def _not_compute_number(self):
        total = 0
        state_list = [TensorState.HOLD, TensorState.HOLD_AFTER_BWD, TensorState.READY_FOR_REDUCE]
        for state in state_list:
            total += self.tensor_state_cnter[state]
        return total

    @property
    def scatter_check(self) -> bool:
        if self.rcache_fused:
            return False
        return self._not_compute_number() == self.num_tensors

    @property
    def reduce_check(self):
        return self.tensor_state_cnter[TensorState.READY_FOR_REDUCE] == self.num_tensors

    def enable_l2_norm_flag(self) -> None:
        self.l2_norm_flag = True

    def set_overflow_flag(self, valid_tensor: torch.Tensor) -> None:
        assert not self.overflow
        self.overflow = torch.isinf(valid_tensor).any().item() | torch.isnan(valid_tensor).any().item()

    def set_l2_norm(self, valid_tensor: torch.Tensor) -> None:
        assert self.l2_norm is None, 'you are calculating the l2 norm twice'
        chunk_l2_norm = valid_tensor.data.float().norm(2)
        self.l2_norm = chunk_l2_norm.item()**2

    def append_tensor(self, tensor: torch.Tensor):
        # sanity check
        assert self.is_init
        assert tensor.dtype == self.chunk_dtype

        new_utilized_size = self.utilized_size + tensor.numel()
        # raise exception when the chunk size is exceeded
        if new_utilized_size > self.chunk_size:
            raise ChunkFullError

        self.chunk_temp[self.utilized_size:new_utilized_size].copy_(tensor.data.flatten())
        tensor.data = self.chunk_temp[self.utilized_size:new_utilized_size].view(tensor.shape)
        fake_data = FakeTensor(tensor.data)

        # record all the information about the tensor
        self.num_tensors += 1
        tensor_state = TensorState.HOLD
        self.tensor_state_cnter[tensor_state] += 1

        self.tensors_info[tensor] = TensorInfo(state=tensor_state,
                                               fake_data=fake_data,
                                               offset=self.utilized_size,
                                               end=new_utilized_size)

        self.utilized_size = new_utilized_size

    def close_chunk(self):
        # sanity check
        assert self.is_init

        # calculate the valid end for each shard
        if self.utilized_size <= self.shard_begin:
            self.valid_end = 0
        elif self.utilized_size < self.shard_end:
            self.valid_end = self.utilized_size - self.shard_begin

        self.__remove_tensors_ptr()
        self.__update_shard(self.chunk_temp, self.shard)
        self.is_replica = False
        self.chunk_temp = None

    def replicate(self):
        assert not self.is_replica

        self.is_replica = True
        this_shard = self.shard if self.optim_sync_flag else self.__paired_shard()
        self.__update_replica(self.rcb.payload, this_shard)
        self.__update_tensors_ptr()

    def scatter(self):
        assert not self.rcache_fused
        assert self.is_replica

        self.__remove_tensors_ptr()
        if not self.optim_sync_flag:
            self.__update_shard(self.rcb.payload, self.shard)
            self.optim_sync_flag = True
        self.is_replica = False

    def reduce(self, always_fp32: bool = False):
        assert self.is_replica

        self.__remove_tensors_ptr()

        if self.pg_size > 1:
            cast_to_fp32 = False
            if always_fp32 and self.chunk_dtype != torch.float:
                cast_to_fp32 = True
                # cast the payload to fp32
                reduce_buffer = self.rcb.payload.to(dtype=torch.float)
            else:
                # otherwise, use the same payload
                reduce_buffer = self.rcb.payload

            # divide the reduce buffer by the size of the process group
            reduce_buffer /= self.pg_size
            # try to use inplace reduce scatter
            # notice: pytorch does not allow true inplace reduce scatter
            # because pytorch will allocate a continuous memory space for collective communications
            shard_buffer = reduce_buffer[self.shard_begin:self.shard_end]
            dist.reduce_scatter_tensor(shard_buffer, reduce_buffer, group=self.torch_pg)

            # the result should be moved to payload for norm calculating
            if cast_to_fp32:
                calc_buffer = self.rcb.payload[self.shard_begin:self.shard_end]
                calc_buffer.copy_(shard_buffer)
        else:
            # if process group size equals to 1, do not communicate
            reduce_buffer = self.rcb.payload

        self.__update_shard(reduce_buffer, self.shard)

        self.is_replica = False

    def access_chunk(self, block: Optional[TensorBlock] = None):
        # sanity check
        assert not self.is_init
        assert not self.is_replica

        if self.rcache_fused:
            assert block is None
            self.rcb = self._my_block
        else:
            assert block in self.rcache.public_used_blocks
            assert self.rcb is None
            self.rcb = block

        self.replicate()

    def release_chunk(self) -> TensorBlock:
        # sanity check
        assert not self.is_init
        assert self.is_replica

        if self.rcache_fused:
            raise RuntimeError

        self.scatter()
        block = self.rcb
        self.rcb = None
        return block

    def update_extra_reduce_info(self, block: Optional[TensorBlock]):
        if self.rcache_fused:
            assert block is None
            block = self._my_block
        else:
            assert block is not None

        buffer = block.payload[self.shard_begin:self.shard_end]
        valid_tensor = buffer[:self.valid_end]
        self.set_overflow_flag(valid_tensor)
        if self.l2_norm_flag:
            self.set_l2_norm(valid_tensor)

    def reduce_chunk(self, always_fp32: bool = False, sync: bool = True) -> Optional[TensorBlock]:
        """Reduce scatter all the gradients. It's an operation done in CUDA.
        """
        # sanity check
        assert not self.is_init
        assert self.is_replica

        self.reduce(always_fp32=always_fp32)
        self.__update_tensors_state(TensorState.HOLD)

        # reset the rcb pointer
        block = self.rcb
        self.rcb = None
        if self.rcache_fused:
            block = None

        if sync:
            self.update_extra_reduce_info(block)

        return block

    def tensor_trans_state(self, tensor: torch.Tensor, tensor_state: TensorState) -> None:
        prev_state = self.tensors_info[tensor].state
        if prev_state == tensor_state:
            return

        # validate whether the update is legal
        # if illegal, raise an exception
        is_update_valid = validate_tensor_state_update(prev_state, tensor_state, raise_exception=True)
        if is_update_valid:
            self.__update_one_tensor_info(self.tensors_info[tensor], tensor_state)

    def copy_tensor_to_chunk_slice(self, tensor: torch.Tensor, data_slice: torch.Tensor) -> None:
        # sanity check
        assert self.is_replica

        info = self.tensors_info[tensor]
        payload = self.rcb.payload
        payload[info.offset:info.end].copy_(data_slice.data.flatten())
        tensor.data = payload[info.offset:info.end].view(tensor.shape)

    def init_pair(self, friend_chunk: 'Chunk') -> None:
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
        friend_chunk: Chunk = self.paired_chunk
        assert not friend_chunk.is_replica
        # gradient and optimizer should be on the same device
        assert self.shard_device.type == friend_chunk.shard_device.type

        if self.shard_device.type == 'cuda':
            self.shard.copy_(friend_chunk.shard)
            self.optim_sync_flag = True
        elif self.shard_device.type == 'cpu':
            # optim_sync_flag is set to False
            # see shard_move function for more details
            self.optim_sync_flag = False
        else:
            raise NotImplementedError

    def get_tensors(self) -> List[torch.Tensor]:
        return list(self.tensors_info.keys())

    def get_cpu_copy(self, only_rank_0: bool = False) -> List[torch.Tensor]:
        assert not self.is_init

        if self.is_replica:
            # use the payload directly when being replica
            temp_buffer = self.rcb.payload
        else:
            # otherwise, create a temporary buffer
            temp_buffer = torch.empty(self.chunk_size, dtype=self.chunk_dtype, device=gpu_device())
            # cheat the assertion in __update_replica
            self.is_replica = True
            self.__update_replica(temp_buffer, self.shard)
            self.is_replica = False

        cpu_copys = [None] * self.num_tensors
        if not only_rank_0 or self.pg_rank == 0:
            for i, (t, info) in enumerate(self.tensors_info.items()):
                t_copy = temp_buffer[info.offset:info.end].view(t.shape).cpu()
                cpu_copys[i] = t_copy
        # synchronize
        dist.barrier()
        return cpu_copys

    def load_tensors(self, tensor_list: List[Optional[torch.Tensor]], only_rank_0: bool = False) -> bool:
        assert not self.is_replica
        assert not self.is_init
        temp_buffer = torch.empty(self.chunk_size, dtype=self.chunk_dtype, device=gpu_device())
        # cheat the assertion in __update_replica
        self.is_replica = True
        self.__update_replica(temp_buffer, self.shard)
        self.is_replica = False

        if not only_rank_0 or self.pg_rank == 0:
            for (_, c_info), load_tensor in zip(self.tensors_info.items(), tensor_list):
                if load_tensor is None:
                    continue
                temp_buffer[c_info.offset:c_info.end].copy_(load_tensor.data.flatten())

        # synchronize
        dist.barrier()

        if only_rank_0:
            dist.broadcast(temp_buffer, src=0, group=self.torch_pg)

        # cheat the assertion in __update_shard
        self.is_replica = True
        self.__update_shard(temp_buffer, self.shard)
        self.is_replica = False

    def __update_replica(self, replica: torch.Tensor, shard: torch.Tensor):
        assert self.is_replica
        assert replica.numel() == self.chunk_size
        assert shard.numel() == self.shard_size

        buffer = replica[self.shard_begin:self.shard_end]
        buffer.copy_(shard)
        dist.all_gather_into_tensor(replica, buffer, group=self.torch_pg)

    def __update_shard(self, replica: torch.Tensor, shard: torch.Tensor):
        assert self.is_replica
        assert replica.numel() == self.chunk_size
        assert shard.numel() == self.shard_size

        shard.copy_(replica[self.shard_begin:self.shard_end])

    def __paired_shard(self):
        assert self.paired_chunk is not None, 'chunks should be paired before training'
        optim_chunk: Chunk = self.paired_chunk
        assert self.chunk_size == optim_chunk.chunk_size

        # only be called when optimizer state is in CPU memory
        # the grad and param should be in the same device
        assert self.shard_device.type == 'cpu'
        return optim_chunk.shard.to(gpu_device())

    def __remove_tensors_ptr(self) -> None:
        # sanity check
        # each tensor should point to its fake data before scatter
        assert self.is_replica
        for tensor, info in self.tensors_info.items():
            tensor.data = info.fake_data

    def __update_tensors_ptr(self) -> None:
        # sanity check
        # the chunk should be replicated to get the correct pointer
        assert self.is_replica
        payload = self.rcb.payload
        for tensor, info in self.tensors_info.items():
            tensor.data = payload[info.offset:info.end].view(tensor.shape)

    def __update_one_tensor_info(self, tensor_info: TensorInfo, next_state: TensorState):
        self.tensor_state_cnter[tensor_info.state] -= 1
        tensor_info.state = next_state
        self.tensor_state_cnter[tensor_info.state] += 1

    def __update_tensors_state(self, next_state: TensorState, prev_state: Optional[TensorState] = None):
        for tensor_info in self.tensors_info.values():
            if prev_state is None or tensor_info.state == prev_state:
                self.__update_one_tensor_info(tensor_info, next_state)

    def __hash__(self) -> int:
        return self.chunk_id

    def __lt__(self, other: object) -> bool:
        return self.chunk_id < other.chunk_id

    def __eq__(self, other: object) -> bool:
        return self.chunk_id == other.chunk_id

    def __repr__(self, detailed: bool = True):
        if self.is_init:
            state = 'initialization'
        elif self.in_rcache:
            state = 'replicated'
        else:
            state = 'scattered'

        output = [
            f'Chunk {self.chunk_id} details: state -> {state}\n',
            f'  length: {self.chunk_size}, dtype: {self.chunk_dtype}, group_size: {self.pg_size}, tensors: {self.num_tensors}\n'
            f'  utilized size: {self.utilized_size}, utilized percentage: {100 * (self.utilized_size / self.chunk_size):.0f}%\n'
        ]

        memory_info = self.memory_usage
        output.append('  memory usage: (cuda -> {}, cpu -> {})\n'.format(memory_info['cuda'], memory_info['cpu']))

        def print_tensor(name, tensor, prefix=''):
            output.append(f'{prefix}{name}: (shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device})\n')

        if self.is_init:
            print_tensor(name='temp', tensor=self.chunk_temp, prefix='  ')
        if self.in_rcache:
            print_tensor(name='block', tensor=self.rcb.payload, prefix='  ')
        if self.shard is not None:
            print_tensor(name='shard', tensor=self.shard, prefix='  ')

        if detailed:
            output.append('  tensor state monitor:\n')
            for st in TensorState:
                output.append('    # of {}: {}\n'.format(st, self.tensor_state_cnter[st]))

        return ''.join(output)
