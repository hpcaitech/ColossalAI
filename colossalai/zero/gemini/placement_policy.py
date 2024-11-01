import functools
import warnings
from abc import ABC, abstractmethod
from time import time
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.distributed as dist

from colossalai.accelerator import get_accelerator
from colossalai.zero.gemini.chunk import Chunk

from .chunk import Chunk, ChunkManager
from .memory_tracer import ChunkMemStatsCollector


class PlacementPolicy(ABC):
    need_mem_stats: bool = False

    def __init__(
        self,
        chunk_manager: ChunkManager,
        mem_stats_collector: Optional[ChunkMemStatsCollector] = None,
        max_prefetch: int = 0,
        **kwargs,
    ) -> None:
        self.chunk_manager = chunk_manager
        self.mem_stats_collector: Optional[ChunkMemStatsCollector] = mem_stats_collector
        self.max_prefetch = max_prefetch

    @abstractmethod
    def evict_tensors(self, can_evict_chunks: List[Chunk], **kwargs) -> Tuple[int, float]:
        raise NotImplementedError

    @abstractmethod
    def setup_grads_device(
        self, params: List[torch.Tensor], grads_device_map: Dict[torch.Tensor, torch.device]
    ) -> None:
        raise NotImplementedError

    def get_prefetch_chunks(
        self, is_warmup, compute_list: tuple, compute_idx: int, async_works: Dict[Chunk, dist.Work]
    ) -> List[Chunk]:
        return []  # no prefetch by default


class StaticPlacementPolicy(PlacementPolicy):
    def __init__(
        self,
        chunk_manager: ChunkManager,
        mem_stats_collector: Optional[ChunkMemStatsCollector] = None,
        max_prefetch: int = 0,
        shard_param_frac: float = 1.0,
        offload_optim_frac: float = 0.0,
        offload_param_frac: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(chunk_manager, mem_stats_collector=mem_stats_collector, max_prefetch=max_prefetch)
        if offload_param_frac > 0.0 and (shard_param_frac != 1.0 or offload_optim_frac != 1.0):
            warnings.warn("offload_param_frac is ignored when shard_param_frac != 1.0 or offload_optim_frac != 1.0")
            offload_param_frac = 0.0
        self.shard_param_frac = shard_param_frac
        self.offload_optim_frac = offload_optim_frac
        self.offload_param_frac = offload_param_frac
        # these should be initialized in setup_grads_device
        self.keep_gathered_chunk_mem = 0.0
        self.keep_cuda_chunk_mem = 0.0

    def evict_tensors(self, can_evict_chunks: List[Chunk], **kwargs) -> Tuple[int, float]:
        can_shard_chunk_mem = sum(chunk.chunk_mem for chunk in can_evict_chunks)
        can_offload_chunk_mem = can_shard_chunk_mem
        for chunk in can_evict_chunks:
            if can_shard_chunk_mem <= self.keep_gathered_chunk_mem:
                break
            self.chunk_manager.release_chunk(chunk)
            # real saved mem is chunk_mem - shard_mem, for simplicity we use chunk_mem
            can_shard_chunk_mem -= chunk.chunk_mem
        for chunk in can_evict_chunks:
            if can_offload_chunk_mem <= self.keep_cuda_chunk_mem:
                break
            self.chunk_manager.move_chunk(chunk, torch.device("cpu"))
            # real saved mem is shard_mem, for simplicity we use chunk_mem
            can_offload_chunk_mem -= chunk.chunk_mem
        return 0, 0.0

    def setup_grads_device(
        self, params: List[torch.Tensor], grads_device_map: Dict[torch.Tensor, torch.device]
    ) -> None:
        total_chunk_mem = sum(self.chunk_manager.get_chunk(p).chunk_mem for p in params)

        offload_optim_chunk_mem = total_chunk_mem * self.offload_optim_frac
        offloaded_optim_chunk_mem = 0
        chunks = set(self.chunk_manager.get_chunk(p) for p in params)
        for chunk in chunks:
            params = chunk.get_tensors()
            # init offload optim settings
            # keep gathered chunks are in CUDA
            if chunk.keep_gathered or offloaded_optim_chunk_mem >= offload_optim_chunk_mem:
                device = get_accelerator().get_current_device()
            else:
                device = torch.device("cpu")
                # real offloaded mem is chunk.shard_mem, for simplicity we use chunk mem here
                offloaded_optim_chunk_mem += chunk.chunk_mem
            for p in params:
                grads_device_map[p] = device
        self.keep_gathered_chunk_mem = total_chunk_mem * (1 - self.shard_param_frac)
        self.keep_cuda_chunk_mem = total_chunk_mem * (1 - self.offload_param_frac)

    def get_prefetch_chunks(
        self, is_warmup: bool, compute_list: tuple, compute_idx: int, async_works: Dict[Chunk, dist.Work]
    ) -> List[Chunk]:
        if is_warmup:  # no prefetch during warmup since we need compute_list
            return []
        can_prefetch = self.max_prefetch - len(async_works)
        prefetch = []
        for i in range(compute_idx + 1, len(compute_list)):
            for chunk in compute_list[i]:
                if len(prefetch) >= can_prefetch:
                    break
                if chunk not in prefetch and chunk not in self.chunk_manager.accessed_chunks:
                    prefetch.append(chunk)
            else:
                continue
            break
        return prefetch


class AutoPlacementPolicy(PlacementPolicy):
    need_mem_stats: bool = True

    def __init__(
        self,
        chunk_manager: ChunkManager,
        mem_stats_collector: Optional[ChunkMemStatsCollector] = None,
        max_prefetch: int = 0,
        warmup_non_model_data_ratio: float = 0.8,
        steady_cuda_cap_ratio: float = 0.9,
        **kwargs,
    ) -> None:
        super().__init__(chunk_manager, mem_stats_collector=mem_stats_collector, max_prefetch=max_prefetch)
        # model data will use 1-_warmup_non_model_data_ratio CUDA memory in warmup phase
        # you can set them by AutoPlacementPolicy.set_warmup_non_model_data_ratio()
        # and AutoPlacementPolicy.set_steady_cuda_cap_ratio()
        self._warmup_non_model_data_ratio = warmup_non_model_data_ratio
        self._steady_cuda_cap_ratio = steady_cuda_cap_ratio

        self.__avail_cuda_model_data_for_prefetch = None

    def evict_tensors(
        self,
        can_evict_chunks: List[Chunk],
        cuda_demand: int = 0,
        warmup: bool = True,
        compute_list: Optional[List[Tuple[Chunk, ...]]] = None,
        compute_idx: int = 0,
        **kwargs,
    ) -> Tuple[int, float]:
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
        from colossalai.legacy.utils.memory import colo_device_memory_capacity

        start = time()
        cuda_capacity = colo_device_memory_capacity(get_accelerator().get_current_device())
        used_cuda_model_data = self.chunk_manager.total_mem["cuda"]
        if warmup:
            # We designate a part of CUDA memory for model data in warmup iterations.
            max_cuda_non_model_data_per_period = cuda_capacity * self._warmup_non_model_data_ratio
        else:
            # max non-model-data cuda memory consumption of this sampling moment and the next sampling moment.
            max_cuda_non_model_data_per_period = self.mem_stats_collector.next_period_non_model_data_usage("cuda")
            cuda_capacity *= self._steady_cuda_cap_ratio
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
                self.chunk_manager.move_chunk(chunk, torch.device("cpu"))
                freed_cuda_model_data += chunk.chunk_mem
            if freed_cuda_model_data < to_free_cuda_model_data:
                raise RuntimeError(
                    f"Adjust layout failed! No enough CUDA memory! "
                    f"Need {to_free_cuda_model_data}, freed {freed_cuda_model_data}"
                )
        self.__avail_cuda_model_data_for_prefetch = avail_cuda_model_data + freed_cuda_model_data
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

    def setup_grads_device(
        self, params: List[torch.Tensor], grads_device_map: Dict[torch.Tensor, torch.device]
    ) -> None:
        for p in params:
            chunk = self.chunk_manager.get_chunk(p)
            # init offload optim settings
            # keep gathered chunks are in CUDA
            if chunk.keep_gathered:
                grads_device_map[p] = get_accelerator().get_current_device()
            else:
                grads_device_map[p] = torch.device("cpu")

    def get_prefetch_chunks(
        self, is_warmup: bool, compute_list: tuple, compute_idx: int, async_works: Dict[Chunk, dist.Work]
    ) -> List[Chunk]:
        if is_warmup:  # no prefetch during warmup since we need compute_list
            return []

        avail_cuda_model_data = self.__avail_cuda_model_data_for_prefetch
        self.__avail_cuda_model_data_for_prefetch = None  # incase of double use

        prefetch_chunk_memory = 0
        can_prefetch = self.max_prefetch - len(async_works)
        prefetch = []
        for i in range(compute_idx + 1, len(compute_list)):
            for chunk in compute_list[i]:
                if len(prefetch) >= can_prefetch or prefetch_chunk_memory + chunk.chunk_mem > avail_cuda_model_data:
                    break
                if chunk not in prefetch and chunk not in self.chunk_manager.accessed_chunks:
                    prefetch_chunk_memory += chunk.chunk_mem
                    prefetch.append(chunk)
            else:
                continue
            break
        return prefetch


class PlacementPolicyFactory:
    policies: Dict[str, Type[PlacementPolicy]] = {
        "auto": AutoPlacementPolicy,
        "static": StaticPlacementPolicy,
    }

    @staticmethod
    def create(policy_name: str) -> Type[PlacementPolicy]:
        if policy_name not in PlacementPolicyFactory.policies:
            raise TypeError(f"Unknown tensor placement policy {policy_name}")
        return PlacementPolicyFactory.policies[policy_name]

    @staticmethod
    def get_policy_names():
        return tuple(PlacementPolicyFactory.policies.keys())
