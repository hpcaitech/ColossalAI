import torch
from .memory_tracer.memstats_collector import MemStatsCollectorV2
from typing import List, Optional, Tuple
from time import time
from colossalai.tensor.chunk import Chunk, ChunkManager
from .placement_policy import PlacementPolicy, PlacementPolicyFactory


class GeminiManager:
    """
    Stateful Tensor Manager, inspired from PatrickStar

    PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management
    https://arxiv.org/abs/2108.05818
    """

    def __init__(self, placement_policy: str, chunk_manager: ChunkManager) -> None:
        assert placement_policy in PlacementPolicyFactory.get_polocy_names()
        policy_cls = PlacementPolicyFactory.create(placement_policy)
        self._chunk_manager = chunk_manager
        self._mem_stats_collector = MemStatsCollectorV2(chunk_manager) if policy_cls.need_mem_stats else None
        self._placement_policy = policy_cls(chunk_manager, self._mem_stats_collector)
        self._compute_list: List[Tuple[Chunk, ...]] = []
        self._compute_idx: int = -1

        self._cpu_gpu_move_volume = 0
        self._layout_time = 0
        self._evict_time = 0
        self._warmup = True

    def pre_iter(self):
        if self._mem_stats_collector and self._warmup:
            self._mem_stats_collector.start_collection()

    def post_iter(self):
        """This function must be called when each iteration finishes
        """
        if self._mem_stats_collector and self._warmup:
            self._mem_stats_collector.finish_collection()
        self._warmup = False
        self._compute_idx = -1
        self._cpu_gpu_move_volume = 0
        self._layout_time = 0
        self._evict_time = 0

    def adjust_layout(self, chunks: Tuple[Chunk, ...], group_name: str) -> None:
        """ Adjust the layout of statefuil tensor according to the information provided
        by mem_stats_collector, which should belongs to a Sharded Model.
        """
        # find stateful tensor in state COMPUTE
        start = time()
        self._record_chunks_order(chunks)
        cuda_demand, hold_cuda_tensor_list = self._get_layout_info(self._compute_idx, self._warmup, chunks, group_name)
        self._layout_time += time() - start
        vol, evict_time = self._placement_policy.evict_tensors(hold_cuda_tensor_list,
                                                               cuda_demand=cuda_demand,
                                                               warmup=self._warmup,
                                                               compute_list=self._compute_list,
                                                               compute_idx=self._compute_idx)
        self._cpu_gpu_move_volume += vol
        self._evict_time += evict_time
        # move COMPUTE tensors to CUDA
        self._cpu_gpu_move_volume += cuda_demand

    @property
    def cpu_gpu_move_volume(self):
        return self._cpu_gpu_move_volume

    # @functools.lru_cache(maxsize=None)
    # TODO: test lru
    def _get_layout_info(self, compute_idx: int, warmup: bool, chunks: Tuple[Chunk, ...], group_name: str):
        cuda_demand = 0
        for chunk in chunks:
            if chunk.device_type == 'cpu' or chunk.is_empty:
                cuda_demand += chunk.mem
        can_evict_chunks = []
        for chunk in self._chunk_manager.chunk_groups[group_name]:
            if not chunk.is_empty and chunk.device_type == 'cuda' and chunk.can_move_device:
                can_evict_chunks.append(chunk)
        return cuda_demand, can_evict_chunks

    def _record_chunks_order(self, chunks: Tuple[Chunk, ...]) -> None:
        self._compute_idx += 1
        if self._warmup and self._placement_policy.need_mem_stats:
            self._compute_list.append(chunks)

    @property
    def default_device(self):
        return self._placement_policy.get_default_device()

    def sample_overall_data(self):
        if self._mem_stats_collector:
            self._mem_stats_collector.sample_overall_data()

    def sample_model_data(self):
        if self._mem_stats_collector:
            self._mem_stats_collector.sample_model_data()

    @property
    def chunk_manager(self):
        return self._chunk_manager

    @property
    def cuda_margin_mem(self) -> Optional[float]:
        if self._mem_stats_collector:
            return self._mem_stats_collector.cuda_margin_mem
        return None

    @property
    def is_cuda_margin_mem_avail(self) -> bool:
        return self._placement_policy.need_mem_stats

    @staticmethod
    def get_default_device(policy_name: str) -> torch.device:
        return PlacementPolicyFactory.get_default_device(policy_name)
