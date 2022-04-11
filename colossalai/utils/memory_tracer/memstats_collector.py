from colossalai.utils.memory_tracer.model_data_memtracer import GLOBAL_MODEL_DATA_TRACER
from colossalai.utils.memory_utils.utils import colo_device_memory_used
from colossalai.utils.memory_tracer.async_memtracer import AsyncMemoryMonitor
import torch
import time
from typing import List


class MemStatsCollector:
    """
    A Memory statistic collector.
    It works in two phases. 
    Phase 1. Collection Phase: collect memory usage statistics of CPU and GPU.
    The first iteration of DNN training.
    Phase 2. Runtime Phase: use the read-only collected stats
    The rest iterations of DNN training.

    It has a Sampling counter which is reset after DNN training iteration.
    """

    def __init__(self) -> None:
        self._mem_monitor = AsyncMemoryMonitor()
        self._model_data_cuda_list = []
        self._overall_cuda_list = []

        self._model_data_cpu_list = []
        self._overall_cpu_list = []

        self._non_model_data_cuda_list = []
        self._non_model_data_cpu_list = []
        self._sampling_time = []

        self._start_flag = False
        self._period_idx = 0

    def overall_mem_stats(self, device_type: str):
        if device_type == 'cuda':
            return self._overall_cuda_list
        elif device_type == 'cpu':
            return self._overall_cpu_list
        else:
            raise TypeError

    def model_data_list(self, device_type: str, unit: str = 'B') -> List[int]:
        if unit == 'GB':
            scale = 1e9
        elif unit == 'MB':
            scale = 1e6
        elif unit == 'KB':
            scale = 1e3
        elif unit == 'B':
            scale = 1
        else:
            raise TypeError

        if device_type == 'cuda':
            return [elem / scale for elem in self._model_data_cuda_list]
        elif device_type == 'cpu':
            return [elem / scale for elem in self._model_data_cpu_list]
        else:
            raise TypeError

    def non_model_data_list(self, device_type: str, unit: str = 'B') -> List[int]:
        """Non model data stats
        """
        if unit == 'GB':
            scale = 1e9
        elif unit == 'MB':
            scale = 1e6
        elif unit == 'KB':
            scale = 1e3
        elif unit == 'B':
            scale = 1
        else:
            raise TypeError

        if device_type == 'cuda':
            return [elem / scale for elem in self._non_model_data_cuda_list]
        elif device_type == 'cpu':
            return [elem / scale for elem in self._non_model_data_cpu_list]
        else:
            raise TypeError

    def max_non_model_data(self, device_type: str) -> int:
        """Get max non model data memory usage of current sampling period

        Args:
            device_type (str): device type, can be 'cpu' or 'cuda'.

        Returns:
            int: max non model data memory usage of current sampling period
        """
        assert not self._start_flag, 'Cannot get mem stats info during collection phase.'
        assert len(self._sampling_time) > 0, 'Cannot get mem stats info before collection phase.'
        next_period_idx = (self._period_idx + 1) % len(self._sampling_time)
        current_non_model_data = self.non_model_data_list(device_type)[self._period_idx]
        next_non_model_data = self.non_model_data_list(device_type)[next_period_idx]
        self._period_idx = next_period_idx
        return max(current_non_model_data, next_non_model_data)

    @property
    def sampling_time(self):
        return [t - self._sampling_time[0] for t in self._sampling_time]

    def start_collection(self):
        self._start_flag = True
        self._mem_monitor.start()

    def finish_collection(self):
        self._start_flag = False
        self._mem_monitor.finish()

    def sample_memstats(self) -> None:
        """
        Sampling memory statistics.
        Record the current model data CUDA memory usage as well as system CUDA memory usage.
        Advance the sampling cnter.
        """
        if self._start_flag:
            self._model_data_cuda_list.append(GLOBAL_MODEL_DATA_TRACER.cuda_usage)
            self._overall_cuda_list.append(self._mem_monitor.finish())
            self._non_model_data_cuda_list.append(self._model_data_cuda_list[-1] - self._overall_cuda_list[-1])

            self._model_data_cpu_list.append(GLOBAL_MODEL_DATA_TRACER.cpu_usage)
            # FIXME(jiaruifang) cpu sys used should also return from self._mem_monitor()
            self._overall_cpu_list.append(colo_device_memory_used(torch.device(f'cpu')))
            self._non_model_data_cpu_list.append(self._overall_cpu_list[-1] - self._model_data_cpu_list[-1])
            self._sampling_time.append(time.time())
            self._mem_monitor.start()

    def clear(self) -> None:
        self._model_data_cuda_list = []
        self._overall_cuda_list = []

        self._model_data_cpu_list = []
        self._overall_cpu_list = []

        self._start_flag = False
        self._period_idx = 0
