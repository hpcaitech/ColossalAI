from colossalai.gemini.chunk import ChunkManager
from colossalai.utils import get_current_device
from colossalai.utils.memory import colo_device_memory_capacity

from .memstats_collector import MemStatsCollector


class ChunkMemStatsCollector(MemStatsCollector):

    def __init__(self, chunk_manager: ChunkManager) -> None:
        super().__init__()
        self._chunk_manager = chunk_manager

    def sample_model_data(self) -> None:
        """Sampling model data statistics.
        """
        if self._start_flag:
            cuda_mem = self._chunk_manager.total_mem['cuda']
            cpu_mem = self._chunk_manager.total_mem['cpu']
            self._model_data_cuda_list.append(cuda_mem)
            self._model_data_cpu_list.append(cpu_mem)

    @property
    def cuda_margin_mem(self) -> float:
        return colo_device_memory_capacity(get_current_device()) - max(self.overall_mem_stats('cuda'))
