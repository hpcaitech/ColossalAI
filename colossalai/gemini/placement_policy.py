import functools
from abc import ABC, abstractmethod
from time import time
from typing import Dict, List, Optional, Tuple, Type

import torch

from colossalai.gemini.chunk import Chunk, ChunkManager
from colossalai.gemini.memory_tracer import ChunkMemStatsCollector
from colossalai.utils import get_current_device
from colossalai.utils.memory import colo_device_memory_capacity


class PlacementPolicy(ABC):
    need_mem_stats: bool = False

    def __init__(self,
                 chunk_manager: ChunkManager,
                 mem_stats_collector: Optional[ChunkMemStatsCollector] = None) -> None:
        self.chunk_manager = chunk_manager
        self.mem_stats_collector: Optional[ChunkMemStatsCollector] = mem_stats_collector

    @abstractmethod
    def evict_tensors(self, can_evict_chunks: List[Chunk], **kwargs) -> Tuple[int, float]:
        raise NotImplementedError

    @staticmethod
    def get_default_device() -> torch.device:
        return torch.device('cpu')


class CPUPlacementPolicy(PlacementPolicy):

    def __init__(self,
                 chunk_manager: ChunkManager,
                 mem_stats_collector: Optional[ChunkMemStatsCollector] = None) -> None:
        super().__init__(chunk_manager, mem_stats_collector=mem_stats_collector)

    def evict_tensors(self, can_evict_chunks: List[Chunk], **kwargs) -> Tuple[int, float]:
        volume = 0
        start = time()
        for chunk in can_evict_chunks:
            self.chunk_manager.release_chunk(chunk)
            self.chunk_manager.move_chunk(chunk, torch.device('cpu'))
            volume += chunk.chunk_mem
        return volume, time() - start


class CUDAPlacementPolicy(PlacementPolicy):

    def __init__(self,
                 chunk_manager: ChunkManager,
                 mem_stats_collector: Optional[ChunkMemStatsCollector] = None) -> None:
        assert torch.cuda.is_available(), 'Cannot use CUDATensorPlacementPolicy when CUDA is not available'
        super().__init__(chunk_manager, mem_stats_collector=mem_stats_collector)

    def evict_tensors(self, can_evict_chunks: List[Chunk], **kwargs) -> Tuple[int, float]:
        return 0, 0

    @staticmethod
    def get_default_device() -> torch.device:
        return get_current_device()


class AutoPlacementPolicy(PlacementPolicy):

    need_mem_stats: bool = True
    # model data will use 1-_warmup_non_model_data_ratio CUDA memory in warmup phase
    # you can set them by AutoPlacementPolicy.set_warmup_non_model_data_ratio()
    # and AutoPlacementPolicy.set_steady_cuda_cap_ratio()
    _warmup_non_model_data_ratio: float = 0.8
    _steady_cuda_cap_ratio: float = 0.9

    def __init__(self,
                 chunk_manager: ChunkManager,
                 mem_stats_collector: Optional[ChunkMemStatsCollector] = None) -> None:
        super().__init__(chunk_manager, mem_stats_collector=mem_stats_collector)

    def evict_tensors(self,
                      can_evict_chunks: List[Chunk],
                      cuda_demand: int = 0,
                      warmup: bool = True,
                      compute_list: Optional[List[Tuple[Chunk, ...]]] = None,
                      compute_idx: int = 0,
                      **kwargs) -> Tuple[int, float]:
        """
        Evict tensors from CUDA device.

        Args:
            can_evict_chunks (List[StatefulTensor]): the list of tensors that can be evicted.
            cuda_demand (int, optional): the volume of data needed on cuda device. Defaults to 0.
            warmup (bool, optional): a flag indicates whether in the phase of warmup. Defaults to True.
            compute_list (List[StatefulTensor], optional): TODO. Defaults to [].
            compute_idx (int, optional): the idx of computing device. Defaults to 0.

        Raises:
            RuntimeError:

        Returns:
            int: the volume of memory that is evicted
        """
        start = time()
        cuda_capacity = colo_device_memory_capacity(get_current_device())
        used_cuda_model_data = self.chunk_manager.total_mem['cuda']
        if warmup:
            # We designate a part of CUDA memory for model data in warmup iterations.
            max_cuda_non_model_data_per_period = cuda_capacity * AutoPlacementPolicy._warmup_non_model_data_ratio
        else:
            # max non-model-data cuda memory consumption of this sampling moment and the next sampling moment.
            max_cuda_non_model_data_per_period = self.mem_stats_collector.next_period_non_model_data_usage('cuda')
            cuda_capacity *= AutoPlacementPolicy._steady_cuda_cap_ratio
        total_cuda_model_data = cuda_capacity - max_cuda_non_model_data_per_period
        avail_cuda_model_data = total_cuda_model_data - used_cuda_model_data
        freed_cuda_model_data = 0

        if avail_cuda_model_data < cuda_demand:
            # Move cuda_demand - avail_cuda_model_data volume of tensors
            # to_free_cuda_model_data = cuda_demand - avail_cuda_model_data
            to_free_cuda_model_data = cuda_demand - avail_cuda_model_data
            to_free_chunks = can_evict_chunks
            if not warmup:
                to_free_chunks = self._sort_can_evict_chunks(tuple(to_free_chunks), compute_idx, tuple(compute_list))
                # print(self._sort_can_evict_chunks.cache_info())
            for chunk in to_free_chunks:
                if freed_cuda_model_data >= to_free_cuda_model_data:
                    break

                self.chunk_manager.release_chunk(chunk)
                self.chunk_manager.move_chunk(chunk, torch.device('cpu'))
                freed_cuda_model_data += chunk.chunk_mem
            if freed_cuda_model_data < to_free_cuda_model_data:
                raise RuntimeError(f"Adjust layout failed! No enough CUDA memory! "
                                   f"Need {to_free_cuda_model_data}, freed {freed_cuda_model_data}")
        return freed_cuda_model_data, time() - start

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _sort_can_evict_chunks(can_evict_chunks: tuple, compute_idx: int, compute_list: tuple) -> list:
        next_compute_idx = {chunk: len(compute_list) for chunk in can_evict_chunks}
        for i in range(len(compute_list) - 1, compute_idx, -1):
            for chunk in compute_list[i]:
                if chunk in next_compute_idx:
                    next_compute_idx[chunk] = i
        next_compute_idx = sorted(next_compute_idx.items(), key=lambda pair: pair[1], reverse=True)
        return [t for (t, idx) in next_compute_idx]

    @staticmethod
    def set_warmup_non_model_data_ratio(ratio: float) -> None:
        ratio = float(ratio)
        assert 0.0 < ratio < 1.0
        AutoPlacementPolicy._warmup_non_model_data_ratio = ratio

    @staticmethod
    def set_steady_cuda_cap_ratio(ratio: float) -> None:
        ratio = float(ratio)
        assert 0.0 < ratio < 1.0
        AutoPlacementPolicy._steady_cuda_cap_ratio = ratio


class ConstPlacementPolicy(PlacementPolicy):

    need_mem_stats: bool = False
    _accessed_memory_boundary = 512 * 1024**2

    def __init__(self,
                 chunk_manager: ChunkManager,
                 mem_stats_collector: Optional[ChunkMemStatsCollector] = None) -> None:
        super().__init__(chunk_manager, mem_stats_collector=mem_stats_collector)

    def evict_tensors(self,
                      can_evict_chunks: List[Chunk],
                      cuda_demand: int = 0,
                      warmup: bool = True,
                      compute_list: Optional[List[Tuple[Chunk, ...]]] = None,
                      compute_idx: int = 0,
                      **kwargs) -> Tuple[int, float]:
        """
        See the docstrings in the class `AutoPlacementPolicy`.
        """
        start = time()
        used_accessed_memory = self.chunk_manager.accessed_mem
        avail_accessed_memory = ConstPlacementPolicy._accessed_memory_boundary - used_accessed_memory
        freed_accessed_memory = 0

        if avail_accessed_memory < cuda_demand:
            to_free_memory = cuda_demand - avail_accessed_memory
            to_free_chunks = can_evict_chunks

            if not warmup:
                # sort all chunks
                to_free_chunks = self._sort_can_evict_chunks(tuple(to_free_chunks), compute_idx, tuple(compute_list))

            for chunk in to_free_chunks:
                if freed_accessed_memory >= to_free_memory:
                    break

                self.chunk_manager.release_chunk(chunk)
                self.chunk_manager.move_chunk(chunk, torch.device('cpu'))
                freed_accessed_memory += chunk.chunk_mem

            if freed_accessed_memory < to_free_memory:
                raise RuntimeError(f"Adjust layout failed! No enough CUDA memory! "
                                   f"Need {to_free_memory}, freed {freed_accessed_memory}")
        return freed_accessed_memory, time() - start

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _sort_can_evict_chunks(can_evict_chunks: tuple, compute_idx: int, compute_list: tuple) -> list:
        next_compute_idx = {chunk: len(compute_list) for chunk in can_evict_chunks}
        for i in range(len(compute_list) - 1, compute_idx, -1):
            for chunk in compute_list[i]:
                if chunk in next_compute_idx:
                    next_compute_idx[chunk] = i
        next_compute_idx = sorted(next_compute_idx.items(), key=lambda pair: pair[1], reverse=True)
        return [t for (t, idx) in next_compute_idx]

    @staticmethod
    def set_const_memory_boundary(cuda_memory_mb: int) -> None:
        boundary = int(cuda_memory_mb * 1024**2)
        assert boundary > 0
        ConstPlacementPolicy._accessed_memory_boundary = boundary


class PlacementPolicyFactory:
    policies: Dict[str, Type[PlacementPolicy]] = {
        'cpu': CPUPlacementPolicy,
        'cuda': CUDAPlacementPolicy,
        'auto': AutoPlacementPolicy,
        'const': ConstPlacementPolicy
    }

    @staticmethod
    def create(policy_name: str) -> Type[PlacementPolicy]:
        if policy_name not in PlacementPolicyFactory.policies:
            raise TypeError(f"Unknown tensor placement policy {policy_name}")
        return PlacementPolicyFactory.policies[policy_name]

    @staticmethod
    def get_policy_names():
        return tuple(PlacementPolicyFactory.policies.keys())

    @staticmethod
    def get_default_device(policy_name: str) -> torch.device:
        policy_cls = PlacementPolicyFactory.create(policy_name)
        return policy_cls.get_default_device()
