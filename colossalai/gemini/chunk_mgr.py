import torch
import numpy as np
from typing import Optional, Dict, Deque, Set, List, Tuple, Iterable
from collections import deque

from colossalai.utils import get_current_device
from colossalai.tensor import ProcessGroup as ColoProcessGroup, ColoTensor
from .chunk import Chunk, ChunkFullError, TensorState


class ChunkManager:
    """
    A manager class to manipulate the tensors in chunks.

    Args:
        chunk_size (int): the size of a chunk.
        process_group (ColoProcessGroup): process group of the chunk.
        enable_distributed_storage (bool): optional, allow for distributed storage of a chunk. The default is false.
        init_device (torch.device): optional, the device on which the chunk is initialized. The default is None.
    """

    def __init__(self,
                 chunk_size: Optional[int],
                 process_group: ColoProcessGroup,
                 enable_distributed_storage: bool = False,
                 init_device: Optional[torch.device] = None) -> None:
        assert chunk_size is None or chunk_size > 0
        assert isinstance(process_group, ColoProcessGroup)
        self.chunk_size = chunk_size
        self.process_group = process_group
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
        if isinstance(tensor, ColoTensor):
            assert tensor.get_process_group().dp_process_group() == self.process_group.dp_process_group(
            ), f"Chunk Manager can only manage ColoTensor with the same DP process group"
        try:
            # append the tensor to the last chunk
            self.chunk_groups[group_name][-1].append(tensor)
        except (IndexError, ChunkFullError):
            # the except statement will be triggered when there is no chunk or
            # the last chunk in the chunk group is full
            # this will create a new chunk and allocate this chunk to its corresponding process
            if self.chunk_size is not None and tensor.numel() > self.chunk_size:
                chunk_size = tensor.numel()
            else:
                chunk_size = self.chunk_size or tensor.numel()
            src_rank = self._get_next_src_rank(group_name)
            chunk = Chunk(chunk_size,
                          src_rank,
                          self.process_group,
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
            return self.process_group.dp_local_rank()
        if self.chunk_size is None:
            if group_name not in self.rank_load:
                self.rank_load[group_name] = torch.zeros(self.process_group.dp_world_size(), dtype=torch.int64)

            # the process owning the tensor will be the process with the smallest number of elements
            src_rank = torch.argmin(self.rank_load[group_name]).item()
        else:
            # chunk is owned by processes in a round-robin fashion
            chunk_idx = len(self.chunk_groups[group_name])
            src_rank = chunk_idx % self.process_group.dp_world_size()
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
        if chunk.device_type == device.type:
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
        msg = f'Rank {self.process_group.dp_local_rank()}:\n'
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
                          min_chunk_size: Optional[int] = None,
                          filter_exlarge_params: bool = True) -> int:
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
        if filter_exlarge_params:
            params_numel = _filter_exlarge_params(params_numel)
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


def _filter_exlarge_params(params_numel: List[int]) -> List[int]:
    params_numel_arr = np.array(params_numel)
    std = np.std(params_numel_arr)
    mean = np.mean(params_numel_arr)
    upper_limit = mean + 3 * std
    return list(filter(lambda x: x <= upper_limit, params_numel))
