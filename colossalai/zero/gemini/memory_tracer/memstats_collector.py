import time
from typing import Optional

from .memory_monitor import SyncCudaMemoryMonitor
from .memory_stats import MemStats


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

    def __init__(self, memstats: Optional[MemStats] = None) -> None:
        self._mem_monitor = SyncCudaMemoryMonitor()
        self._sampling_time = []

        self._start_flag = False
        self._step_idx = 0
        self._step_total = 0
        if memstats is not None:
            self.use_outside_memstats = True
            self._memstats = memstats
        else:
            self.use_outside_memstats = False
            self._memstats = MemStats()

    def next_period_non_model_data_usage(self, device_type: str) -> int:
        """Maximum non model data memory usage during the next Op run

        Args:
            device_type (str): device type, can be 'cpu' or 'cuda'.

        Returns:
            int: max non model data memory usage of current sampling period
        """
        assert not self._start_flag, "Cannot get mem stats info during collection phase."
        assert self._step_total > 0, "Cannot get mem stats info before collection phase."
        assert len(self._memstats.non_model_data_list(device_type)) > self._step_idx, (
            f"{len(self._memstats.non_model_data_list(device_type))} should be > than step idx {self._step_idx}, "
            f"step total {self._step_total}"
        )
        next_non_model_data = self._memstats.non_model_data_list(device_type)[self._step_idx]
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
        # self._step_total = len(self._sampling_time)
        self._step_total = len(self._memstats.non_model_data_list("cuda"))
        self._start_flag = False
        print(f"finish_collection {self._step_total}")

    # deprecated
    def record_model_data_volume(self) -> None:
        """
        Sampling model data statistics.
        """
        if self._start_flag and not self.use_outside_memstats:
            from colossalai.legacy.zero.gemini import StatefulTensor

            # The following code work for ZeroInitContext, which is deprecated in v0.1.12
            cuda_mem = StatefulTensor.GST_MGR.total_mem["cuda"]
            self._memstats.record_max_cuda_model_data(cuda_mem)

    def sample_overall_data(self) -> None:
        """
        Sampling overall and non model data cuda memory statistics.
        """
        if self._start_flag and not self.use_outside_memstats:
            cuda_overall = self._mem_monitor.finish()
            self._memstats.record_max_cuda_overall_data(cuda_overall)
            self._memstats.calc_max_cuda_non_model_data()

            self._mem_monitor.start()

        if self._start_flag:
            self._sampling_time.append(time.time())

    def clear(self) -> None:
        self._memstats.clear()
        self._start_flag = False
        self._step_idx = 0
        self._step_total = 0
