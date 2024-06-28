from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.accelerator import get_accelerator
from colossalai.utils import free_storage

from .chunk import Chunk, ChunkFullError, TensorState


class ChunkManager:
    """
    A manager class to manipulate the tensors in chunks.

    Args:
        chunk_configuration (Dict[int, Dict]): the configuration dictionary of this chunk manager.
        init_device (torch.device): optional, the device on which the chunk is initialized. The default is None.
    """

    def __init__(
        self,
        chunk_configuration,
        init_device: Optional[torch.device] = None,
        reuse_fp16_chunk: bool = True,
        max_prefetch: int = 0,
    ) -> None:
        self.device = init_device or get_accelerator().get_current_device()
        self.dp_degree_chunk_size_dict: Dict[int, int] = dict()
        self.kwargs_config = chunk_configuration
        for k, v in self.kwargs_config.items():
            self.dp_degree_chunk_size_dict[k] = v.pop("chunk_size")
            v["init_device"] = self.device

        self.chunk_groups: Dict[str, Deque[Chunk]] = dict()
        self.tensor_chunk_map: Dict[torch.Tensor, Chunk] = dict()
        self.accessed_chunks: Set[Chunk] = set()
        self.accessed_mem: int = 0
        self.total_mem: Dict[str, int] = {"cpu": 0, "cuda": 0}
        self.reuse_fp16_chunk = reuse_fp16_chunk
        # Whether model is accumulating gradients,
        self.accumulating_grads = False
        self.overflow_counter = torch.tensor([0], dtype=torch.int, device=get_accelerator().get_current_device())
        self._prefetch_stream = get_accelerator().Stream() if max_prefetch else None

    def register_tensor(
        self,
        tensor: torch.Tensor,
        group_type: str,
        config_key: int,
        zero_group: ProcessGroup,
        extra_dp_group: ProcessGroup = None,
        cpu_offload: bool = False,
        pin_memory: bool = False,
    ) -> None:
        """
        Register a tensor to the chunk manager.
        Then, the tensor should be accessed by `get_chunks`.

        Args:
            tensor: the tensor appended to the chunk
            group_type: the data type of the group.
            config_key: the key of the group's name, the size of the dp world
            cpu_offload: if True, the chunk will be closed on CPU
            pin_memory: whether the chunk is pinned in the cpu memory
        """
        assert tensor not in self.tensor_chunk_map
        assert isinstance(tensor, torch.Tensor), "Please feed Tensor to this ChunkManager"
        assert config_key in self.dp_degree_chunk_size_dict

        chunk_size = self.dp_degree_chunk_size_dict[config_key]
        chunk_kwargs = self.kwargs_config[config_key]
        group_name = "{}_{}".format(group_type, config_key)
        chunk_group = self.__get_chunk_group(group_name)

        try:
            # append the tensor to the last chunk
            chunk_group[-1].append_tensor(tensor)
        except (IndexError, ChunkFullError):
            # the except statement will be triggered when there is no chunk or
            # the last chunk in the chunk group is full
            # this will create a new chunk and allocate this chunk to its corresponding process
            if chunk_group:
                # the chunk group is not empty
                # close the last chunk
                self.__close_one_chunk(chunk_group[-1])

            if tensor.numel() > chunk_size:
                chunk_size = tensor.numel()
                dp_size = dist.get_world_size(zero_group)
                chunk_size = chunk_size + (-chunk_size % dp_size)

            chunk = Chunk(
                chunk_size=chunk_size,
                zero_group=zero_group,
                dtype=tensor.dtype,
                cpu_shard_init=cpu_offload,
                pin_memory=pin_memory,
                extra_dp_group=extra_dp_group,
                **chunk_kwargs,
            )

            chunk_group.append(chunk)
            chunk.append_tensor(tensor)
            self.__add_memory_usage(chunk.memory_usage)

        self.tensor_chunk_map[tensor] = chunk_group[-1]

    def close_all_groups(self):
        """Close all the chunks of all groups."""
        for group_name in self.chunk_groups:
            self.__close_one_chunk(self.chunk_groups[group_name][-1])

    def access_chunk(self, chunk: Chunk, async_access: bool = False) -> Optional[dist.Work]:
        """Make the chunk can be used for calculation."""
        if chunk in self.accessed_chunks:
            return None
        self.__sub_memory_usage(chunk.memory_usage)
        if chunk.device_type == "cpu":
            chunk.shard_move(get_accelerator().get_current_device(), non_blocking=async_access)
        maybe_work = self.__add_accessed_chunk(chunk, async_access=async_access)
        self.__add_memory_usage(chunk.memory_usage)
        return maybe_work

    def release_chunk(self, chunk: Chunk) -> None:
        """Scatter the chunk in CUDA."""
        if chunk not in self.accessed_chunks:
            return
        if chunk.can_release:
            self.__sub_memory_usage(chunk.memory_usage)
            self.__sub_accessed_chunk(chunk)
            self.__add_memory_usage(chunk.memory_usage)

    def move_chunk(self, chunk: Chunk, device: torch.device, force_copy: bool = False, async_move=False) -> None:
        """Move the shard of the chunk to the target device."""
        if not chunk.can_move or chunk.device_type == device.type:
            return
        self.__sub_memory_usage(chunk.memory_usage)
        chunk.shard_move(device, force_copy, non_blocking=async_move)
        self.__add_memory_usage(chunk.memory_usage)

    def trans_tensor_state(self, tensor: torch.Tensor, state: TensorState) -> None:
        """Transit tensor state according to pre-defined state machine."""
        chunk = self.tensor_chunk_map[tensor]
        chunk.tensor_trans_state(tensor, state)

    def reduce_chunk(self, chunk: Chunk, async_op: bool = False) -> bool:
        """Reduce or all reduce the chunk."""
        if not chunk.can_reduce:
            return False
        self.__sub_memory_usage(chunk.memory_usage)
        chunk.reduce(async_op=async_op)
        self.__sub_accessed_chunk(chunk)
        self.__add_memory_usage(chunk.memory_usage)
        return True

    def fake_release_chunk(self, chunk: Chunk) -> None:
        """Release gathered chunk in a fake mode.
        This function is used for keep-gathered chunk in the inference mode.
        """
        assert chunk.keep_gathered
        assert chunk.tensor_state_cnter[TensorState.HOLD] == chunk.num_tensors
        self.__sub_accessed_chunk(chunk)

    def copy_tensor_to_chunk_slice(self, tensor: torch.Tensor, data: torch.Tensor) -> None:
        """
        Copy data to the chunk.

        Args:
            tensor (torch.Tensor): the tensor used to retrieve meta information
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

    def get_cuda_movable_chunks(self) -> List[Chunk]:
        """
        Get all chunks that can be moved.
        """
        chunk_list = []
        for chunk in self.accessed_chunks:
            if chunk.can_release:
                chunk_list.append(chunk)
        chunk_list.sort(key=lambda x: x.count_id)
        return chunk_list

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
        device_type = tensor.device.type
        if device_type == "npu":
            device_type = "cuda"
        self.total_mem[device_type] += tensor.numel() * tensor.element_size()

    def __repr__(self) -> str:
        msg = [
            "Chunk Manager Information:\n",
            "Total memory: " + ", ".join([f"{k}={v}B" for k, v in self.total_mem.items()]) + "\n",
        ]
        for group_name, group in self.chunk_groups.items():
            msg.append(f"Group {group_name}:\n")
            for i, chunk in enumerate(group):
                msg.append(f"[{i}] {chunk}\n")
        return "".join(msg)

    def __get_chunk_group(self, group_name: str) -> Deque[Chunk]:
        """Register a chunk group."""
        if group_name not in self.chunk_groups:
            self.chunk_groups[group_name] = deque()
        return self.chunk_groups[group_name]

    def __close_one_chunk(self, chunk: Chunk):
        self.__sub_memory_usage(chunk.memory_usage)
        chunk.close_chunk()
        self.__add_memory_usage(chunk.memory_usage)

    def __sub_memory_usage(self, usage: Dict[str, int]):
        for k, v in usage.items():
            self.total_mem[k] -= v

    def __add_memory_usage(self, usage: Dict[str, int]):
        for k, v in usage.items():
            self.total_mem[k] += v

    def __add_accessed_chunk(self, chunk: Chunk, async_access: bool = False) -> Optional[dist.Work]:
        maybe_work = chunk.access_chunk(async_access=async_access)
        self.accessed_chunks.add(chunk)
        self.accessed_mem += chunk.chunk_mem
        return maybe_work

    def __sub_accessed_chunk(self, chunk: Chunk):
        chunk.release_chunk()
        self.accessed_chunks.remove(chunk)
        self.accessed_mem -= chunk.chunk_mem

    def init_grad_chunk(self, chunk: Chunk) -> Chunk:
        if chunk.grad_chunk is not None:
            self.__sub_memory_usage(chunk.grad_chunk.memory_usage)
        grad_chunk = chunk.init_grad_chunk()
        self.__add_memory_usage(grad_chunk.memory_usage)
        if grad_chunk not in self.accessed_chunks:
            self.accessed_chunks.add(grad_chunk)
            self.accessed_mem += grad_chunk.chunk_mem
        return grad_chunk

    def rearrange_accumulated_grad_chunk(self, chunk: Chunk) -> Chunk:
        """Rearrange gradients accumulated in chunk.grad_chunk, and get prepared for gradient reduction."""

        assert chunk.grad_chunk is not None

        # Make a backup for gradient accumulated before.
        # Here backup gradients should be multiplied, since it will be divided after gradient reduction.
        if chunk.grad_chunk.is_gathered:
            accumulated_grad = chunk.grad_chunk.cuda_global_chunk.clone().detach().mul_(chunk.pg_size)
            accumulated_grad_gathered = True
        else:
            if chunk.grad_chunk.cuda_shard is not None:
                accumulated_grad = chunk.grad_chunk.cuda_shard.clone().detach().mul_(chunk.pg_size)
            else:
                accumulated_grad = (
                    chunk.grad_chunk.cpu_shard.to(get_accelerator().get_current_device())
                    .clone()
                    .detach()
                    .mul_(chunk.pg_size)
                )
            accumulated_grad_gathered = False

        # Reset grad_chunk, and chunk.grad_chunk will be accessed.
        grad_chunk = self.init_grad_chunk(chunk)
        grad_chunk.cuda_global_chunk.zero_()

        # Add backup gradients to grad_chunk.
        if accumulated_grad_gathered:
            grad_chunk.cuda_global_chunk.add_(accumulated_grad)
        else:
            grad_chunk.cuda_global_chunk[grad_chunk.shard_begin : grad_chunk.shard_end].add_(accumulated_grad)

        # Release accumulated_grad
        free_storage(accumulated_grad)

        return grad_chunk
