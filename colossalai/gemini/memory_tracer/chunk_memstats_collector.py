from typing import Optional

from colossalai.gemini.chunk import ChunkManager
from colossalai.gemini.memory_tracer import MemStats
from colossalai.utils import get_current_device
from colossalai.utils.memory import colo_device_memory_capacity

from .memstats_collector import MemStatsCollector


class ChunkMemStatsCollector(MemStatsCollector):

    def __init__(self, chunk_manager: ChunkManager, memstats: Optional[MemStats] = None) -> None:
        """

        Memory Statistic Collector for Chunks.

        Args:
            chunk_manager (ChunkManager): the chunk manager.
            memstats (Optional[MemStats], optional): memory statistics collected by RMT. Defaults to None.
        """
        super().__init__(memstats)
        self._chunk_manager = chunk_manager

    # override
    def record_model_data_volume(self) -> None:
        """
        record model data volumn on cuda and cpu.
        """
        if self._start_flag and not self.use_outside_memstats:
            cuda_mem = self._chunk_manager.total_mem['cuda']
            self._memstats.record_max_cuda_model_data(cuda_mem)

    @property
    def cuda_margin_mem(self) -> float:
        return colo_device_memory_capacity(get_current_device()) - self._memstats.max_overall_cuda
