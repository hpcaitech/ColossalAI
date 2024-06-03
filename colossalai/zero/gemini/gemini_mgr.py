import functools
from time import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist

from .chunk import Chunk, ChunkManager
from .memory_tracer import ChunkMemStatsCollector, MemStats
from .placement_policy import PlacementPolicy, PlacementPolicyFactory


class GeminiManager:
    """
    Stateful Tensor Manager, inspired from PatrickStar

    PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management
    https://arxiv.org/abs/2108.05818

    Args:
        placement_policy (str): Which device to place *held* tensors. It can be 'static' and 'auto'.
            If it's 'auto', they are moving dynamically based on CPU and CUDA memory usage. It will utilize heterogeneous memory space evenly and well.
            Note that 'auto' policy can only work well when no other processes use CUDA during your training.
        chunk_manager (ChunkManager): A ``ChunkManager`` instance.
        memstats (MemStats, optional): a mem stats collected by a runtime mem tracer. if None then GeminiManager will collect it during a warmup iteration.
    """

    def __init__(
        self,
        placement_policy: str,
        chunk_manager: ChunkManager,
        memstats: Optional[MemStats] = None,
        **placement_kwargs,
    ) -> None:
        assert placement_policy in PlacementPolicyFactory.get_policy_names()
        self.policy_name = placement_policy
        policy_cls = PlacementPolicyFactory.create(placement_policy)
        self._chunk_manager = chunk_manager

        self._premade_memstats_ = memstats is not None
        self._memstats = memstats
        self._mem_stats_collector = (
            ChunkMemStatsCollector(chunk_manager, self._memstats) if policy_cls.need_mem_stats else None
        )
        self._placement_policy = policy_cls(
            chunk_manager=chunk_manager, mem_stats_collector=self._mem_stats_collector, **placement_kwargs
        )
        self._compute_list: List[Tuple[Chunk, ...]] = []
        self._compute_idx: int = -1
        self._async_works: Dict[Chunk, dist.Work] = {}

        self._h2d_volume = 0
        self._d2h_volume = 0
        self._layout_time = 0
        self._evict_time = 0
        self._warmup = True
        self._comp_cuda_demand_time = 0

    def reset_attributes(self):
        self._compute_idx = -1
        self._h2d_volume = 0
        self._d2h_volume = 0
        self._layout_time = 0
        self._evict_time = 0
        self._comp_cuda_demand_time = 0

    @property
    def need_warmup(self) -> bool:
        return self.policy_name in ("auto", "const")

    def is_warmup(self):
        return self._warmup

    def memstats(self):
        """memstats

        get the memory statistics during training.
        The stats could be collected by a runtime memory tracer, or collected by the GeminiManager.
        Note, for the latter, you can not access the memstats before warmup iteration finishes.
        """
        if self._premade_memstats_:
            return self._memstats
        else:
            assert not self._warmup, "Gemini Manager has memstats after warm up! Now is during warmup."
            return self._mem_stats_collector._memstats

    def pre_iter(self, *args):
        if self._mem_stats_collector and self._warmup:
            self._mem_stats_collector.start_collection()

    def post_iter(self):
        """This function must be called when each iteration finishes"""
        if self._mem_stats_collector and self._warmup:
            self._mem_stats_collector.finish_collection()
        self._warmup = False
        self.reset_attributes()

    def adjust_layout(self, chunks: Tuple[Chunk, ...], record_anyway: bool = False) -> None:
        """Adjust the layout of stateful tensors according to the information provided
        by mem_stats_collector, which should belongs to a Sharded Model.
        """
        # find stateful tensor in state COMPUTE
        start = time()
        self._record_warmup_chunks_order(chunks, record_anyway=record_anyway)
        cuda_demand, can_evict_chunks = self._get_layout_info(self._compute_idx, self._warmup, chunks)
        # don't evict chunks that are asynchronously fetched
        can_evict_chunks = [chunk for chunk in can_evict_chunks if chunk not in self._async_works]
        self._layout_time += time() - start

        vol, evict_time = self._placement_policy.evict_tensors(
            can_evict_chunks=can_evict_chunks,
            cuda_demand=cuda_demand,
            warmup=self._warmup,
            compute_list=self._compute_list,
            compute_idx=self._compute_idx,
        )

        self._d2h_volume += vol
        self._evict_time += evict_time
        # move COMPUTE tensors to CUDA
        self._h2d_volume += cuda_demand

    def wait_chunks(self, chunks: Iterable[Chunk]) -> Tuple[Chunk]:
        non_prefetched_chunks = []
        for chunk in chunks:
            if chunk in self._async_works:
                self._async_works[chunk].wait()
                del self._async_works[chunk]
            else:
                non_prefetched_chunks.append(chunk)
        return tuple(non_prefetched_chunks)

    def add_work(self, chunk: Chunk, work: dist.Work):
        assert work is not None
        assert chunk not in self._async_works
        self._async_works[chunk] = work

    @functools.lru_cache(maxsize=None)
    def _get_layout_info(self, compute_idx: int, warmup: bool, chunks: Tuple[Chunk, ...]):
        start = time()
        cuda_demand = 0
        for chunk in chunks:
            if chunk.device_type == "cuda" or chunk.device_type == "npu":
                if chunk.is_gathered:
                    pass
                else:
                    cuda_demand += chunk.chunk_mem - chunk.shard_mem
            elif chunk.device_type == "cpu":
                cuda_demand += chunk.chunk_mem
            else:
                raise RuntimeError
        self._comp_cuda_demand_time += time() - start

        can_evict_chunks = self._chunk_manager.get_cuda_movable_chunks()
        return cuda_demand, can_evict_chunks

    def _record_warmup_chunks_order(self, chunks: Tuple[Chunk, ...], record_anyway: bool = False) -> None:
        self._compute_idx += 1
        if self._warmup and (self._placement_policy.need_mem_stats or record_anyway):
            self._compute_list.append(chunks)

    def sample_overall_data(self):
        if self._mem_stats_collector:
            self._mem_stats_collector.sample_overall_data()

    def record_model_data_volume(self):
        if self._mem_stats_collector:
            self._mem_stats_collector.record_model_data_volume()

    @property
    def chunk_manager(self):
        return self._chunk_manager

    @property
    def cuda_margin_mem(self) -> Optional[float]:
        if self._mem_stats_collector:
            return self._mem_stats_collector.cuda_margin_mem
        return None

    @property
    def placement_policy(self) -> PlacementPolicy:
        return self._placement_policy

    @property
    def compute_list(self) -> List[Tuple[Chunk, ...]]:
        return self._compute_list

    @property
    def compute_idx(self) -> int:
        return self._compute_idx

    @property
    def async_works(self) -> Dict[Chunk, dist.Work]:
        return self._async_works

    @property
    def is_cuda_margin_mem_avail(self) -> bool:
        return self._placement_policy.need_mem_stats

    def setup_grads_device(
        self, params: List[torch.Tensor], grads_device_map: Dict[torch.Tensor, torch.device]
    ) -> None:
        self._placement_policy.setup_grads_device(params, grads_device_map)
