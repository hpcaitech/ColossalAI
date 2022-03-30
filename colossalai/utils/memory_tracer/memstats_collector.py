from colossalai.utils.memory_tracer.model_data_memtracer import GLOBAL_MODEL_DATA_TRACER
from colossalai.utils.memory_utils.memory_monitor import colo_cuda_memory_used
from colossalai.utils import get_current_device

import torch
from typing import Tuple


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

    def __init__(self) -> None:
        """
        Collecting Memory Statistics.
        It has two phases. 
        1. Collection Phase: collect memory usage statistics
        2. Runtime Phase: do not collect statistics.
        """
        self._sampling_cnter = SamplingCounter()
        self._model_data_cuda = []
        self._overall_cuda = []

        # TODO(jiaruifang) Now no cpu mem stats collecting
        self._model_data_cpu = []
        self._overall_cpu = []

        self._start_flag = False

    @property
    def overall_cuda(self):
        return self._overall_cuda

    @property
    def model_data_cuda_GB(self):
        return [elem / 1e9 for elem in self._model_data_cuda]

    @property
    def model_data_cuda(self):
        return self._model_data_cuda

    @property
    def non_model_data_cuda_GB(self):
        return [elem / 1e9 for elem in self.non_model_data_cuda]

    @property
    def non_model_data_cuda(self):
        """Non model data stats
        """
        return [(v1 - v2) for v1, v2 in zip(self._overall_cuda, self._model_data_cuda)]

    def start_collection(self):
        self._start_flag = True

    def finish_collection(self):
        self._start_flag = False

    def sample_memstats(self) -> None:
        """
        Sampling memory statistics.
        Record the current model data CUDA memory usage as well as system CUDA memory usage.
        """
        if self._start_flag:
            sampling_cnt = self._sampling_cnter.sampling_cnt
            assert sampling_cnt == len(self._overall_cuda)
            self._model_data_cuda.append(GLOBAL_MODEL_DATA_TRACER.cuda_usage)
            self._overall_cuda.append(colo_cuda_memory_used(torch.device(f'cuda:{get_current_device()}')))
        self._sampling_cnter.advance()

    def fetch_memstats(self) -> Tuple[int, int]:
        """
        returns cuda usage of model data and overall cuda usage.
        """
        sampling_cnt = self._sampling_cnter.sampling_cnt
        if len(self._model_data_cuda) < sampling_cnt:
            raise RuntimeError
        return (self._model_data_cuda[sampling_cnt], self._overall_cuda[sampling_cnt])

    def reset_sampling_cnter(self) -> None:
        self._sampling_cnter.reset()

    def clear(self) -> None:
        self._model_data_cuda = []
        self._overall_cuda = []

        self._model_data_cpu = []
        self._overall_cpu = []

        self._start_flag = False
        self._sampling_cnter.reset()
