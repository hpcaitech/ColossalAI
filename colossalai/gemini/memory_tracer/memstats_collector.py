from colossalai.gemini.memory_tracer import SyncCudaMemoryMonitor
from colossalai.utils.memory import colo_device_memory_used
from colossalai.gemini.stateful_tensor import StatefulTensor
from colossalai.tensor import ChunkManager

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
        self._mem_monitor = SyncCudaMemoryMonitor()
        self._model_data_cuda_list = []
        self._overall_cuda_list = []

        self._model_data_cpu_list = []
        self._overall_cpu_list = []

        self._non_model_data_cuda_list = []
        self._non_model_data_cpu_list = []
        self._sampling_time = []

        self._start_flag = False
        self._step_idx = 0
        self._step_total = 0

    def overall_mem_stats(self, device_type: str) -> List[int]:
        if device_type == 'cuda':
            return self._overall_cuda_list
        elif device_type == 'cpu':
            return self._overall_cpu_list
        else:
            raise TypeError

    def model_data_list(self, device_type: str) -> List[int]:
        if device_type == 'cuda':
            return self._model_data_cuda_list
        elif device_type == 'cpu':
            return self._model_data_cpu_list
        else:
            raise TypeError

    def non_model_data_list(self, device_type: str) -> List[int]:
        if device_type == 'cuda':
            return self._non_model_data_cuda_list
        elif device_type == 'cpu':
            return self._non_model_data_cpu_list
        else:
            raise TypeError

    def next_period_non_model_data_usage(self, device_type: str) -> int:
        """Get max non model data memory usage of current sampling period

        Args:
            device_type (str): device type, can be 'cpu' or 'cuda'.

        Returns:
            int: max non model data memory usage of current sampling period
        """
        assert not self._start_flag, 'Cannot get mem stats info during collection phase.'
        assert self._step_total > 0, 'Cannot get mem stats info before collection phase.'
        next_non_model_data = self.non_model_data_list(device_type)[self._step_idx]
        self._step_idx = (self._step_idx + 1) % self._step_total
        return next_non_model_data

    @property
    def sampling_time(self):
        return [t - self._sampling_time[0] for t in self._sampling_time]

    def start_collection(self):
        self._start_flag = True
        self._mem_monitor.start()

    def finish_collection(self):
        self.sample_overall_data()
        self._step_total = len(self._sampling_time)
        self._start_flag = False
        self._mem_monitor.finish()

    def sample_model_data(self) -> None:
        """Sampling model data statistics.
        """
        if self._start_flag:
            cuda_mem = StatefulTensor.GST_MGR.total_mem['cuda']
            cpu_mem = StatefulTensor.GST_MGR.total_mem['cpu']
            self._model_data_cuda_list.append(cuda_mem)
            self._model_data_cpu_list.append(cpu_mem)

    def sample_overall_data(self) -> None:
        """Sampling non model data statistics.
        """
        if self._start_flag:
            # overall data recording is after model data recording
            if len(self._model_data_cuda_list) == 0:
                return

            self._overall_cuda_list.append(self._mem_monitor.finish())
            self._overall_cpu_list.append(colo_device_memory_used(torch.device('cpu')))

            assert len(self._model_data_cuda_list) == len(self._overall_cuda_list)

            self._non_model_data_cuda_list.append(self._overall_cuda_list[-1] - self._model_data_cuda_list[-1])
            self._non_model_data_cpu_list.append(self._overall_cpu_list[-1] - self._model_data_cpu_list[-1])
            self._sampling_time.append(time.time())
            self._mem_monitor.start()

    def clear(self) -> None:
        self._model_data_cuda_list = []
        self._overall_cuda_list = []

        self._model_data_cpu_list = []
        self._overall_cpu_list = []

        self._non_model_data_cpu_list = []
        self._non_model_data_cuda_list = []

        self._start_flag = False
        self._step_idx = 0
        self._step_total = 0


class MemStatsCollectorV2(MemStatsCollector):

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
