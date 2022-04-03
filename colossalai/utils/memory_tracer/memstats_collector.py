from colossalai.utils.memory_tracer.model_data_memtracer import GLOBAL_MODEL_DATA_TRACER
from colossalai.utils.memory_utils.utils import colo_device_memory_used
from colossalai.utils import get_current_device
from colossalai.utils.memory_tracer.async_memtracer import AsyncMemoryMonitor
import torch
import time
from typing import List


class SamplingCounter:

    def __init__(self) -> None:
        self._samplint_cnt = 0

    def advance(self):
        self._samplint_cnt += 1

    @property
    def sampling_cnt(self):
        return self._samplint_cnt

    def reset(self):
        self._samplint_cnt = 0


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
        self._sampling_cnter = SamplingCounter()
        self._mem_monitor = AsyncMemoryMonitor()
        self._model_data_cuda_list = []
        self._overall_cuda_list = []

        self._model_data_cpu_list = []
        self._overall_cpu_list = []

        self._sampling_time = []

        self._start_flag = False

    def overall_mem_stats(self, device_type: str):
        if device_type == 'cuda':
            return self._overall_cuda_list
        elif device_type == 'cpu':
            return self._overall_cpu_list
        else:
            raise TypeError

    def model_data_cuda_list(self, device_type: str, unit: str = 'B') -> List[int]:
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

    def non_model_data_cuda_list(self, device_type: str, unit: str = 'B') -> List[int]:
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
            return [(v1 - v2) / scale for v1, v2 in zip(self._overall_cuda_list, self._model_data_cuda_list)]
        elif device_type == 'cpu':
            return [(v1 - v2) / scale for v1, v2 in zip(self._overall_cpu_list, self._model_data_cpu_list)]
        else:
            raise TypeError

    @property
    def sampling_time(self):
        return [t - self._sampling_time[0] for t in self._sampling_time]

    def start_collection(self):
        self._start_flag = True
        self._mem_monitor.start()

    def finish_collection(self):
        self._start_flag = False

    def sample_memstats(self) -> None:
        """
        Sampling memory statistics.
        Record the current model data CUDA memory usage as well as system CUDA memory usage.
        Advance the sampling cnter.
        """
        if self._start_flag:
            sampling_cnt = self._sampling_cnter.sampling_cnt
            assert sampling_cnt == len(self._overall_cuda_list)
            self._model_data_cuda_list.append(GLOBAL_MODEL_DATA_TRACER.cuda_usage)
            self._overall_cuda_list.append(self._mem_monitor.finish())

            self._model_data_cpu_list.append(GLOBAL_MODEL_DATA_TRACER.cpu_usage)

            # FIXME() cpu sys used should also return from self._mem_monitor()
            self._overall_cpu_list.append(colo_device_memory_used(torch.device(f'cpu')))

            self._sampling_time.append(time.time())
            self._mem_monitor.start()
        self._sampling_cnter.advance()

    def reset_sampling_cnter(self) -> None:
        self._sampling_cnter.reset()
        self._mem_monitor.finish()

    def clear(self) -> None:
        self._model_data_cuda_list = []
        self._overall_cuda_list = []

        self._model_data_cpu_list = []
        self._overall_cpu_list = []

        self._start_flag = False
        self._sampling_cnter.reset()
        self._mem_monitor.finish()