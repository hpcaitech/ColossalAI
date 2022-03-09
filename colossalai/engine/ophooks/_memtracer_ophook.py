from colossalai.context.parallel_mode import ParallelMode
import torch
from . import BaseOpHook
from concurrent.futures import ThreadPoolExecutor
from colossalai.registry import OPHOOKS
from colossalai.logging import get_dist_logger
from time import sleep, time
import pickle
from typing import Optional
from colossalai.core import global_context as gpc
import math


def get_cuda_memory_used(device: Optional[torch.device]) -> int:
    """Get the free memory info of device.
    Notice that for CPU, this function will return 1/N of the total free memory,
    where N is the world size.

    :param device: device id
    :type device: torch.device
    :return: current memory usage, sized by MB
    :rtype: int
    """
    ret: int = torch.cuda.memory_allocated(device)
    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):    # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats(device)
    return ret


class AsyncMemoryMonitor:
    """
        An Async Mem Monitor runing during computing. Sampling GPU memory usage of the current GPU
        at interval of 1/(10**power) sec.

        :param power: the power of time interval, defaults to 10
        :type power: int
        """

    def __init__(self, power: int = 10):

        self.keep_measuring = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.monitor_thread = None
        self.interval = 1 / (10**power)
        self.time_stamps = []
        self.mem_stats = []

    def __len__(self):
        return len(self.mem_stats)

    def set_interval(self, power: int):
        self.clear()
        self.interval = 1 / (10**power)

    def is_measuring(self):
        return self.keep_measuring

    def start(self):
        self.keep_measuring = True
        self.monitor_thread = self.executor.submit(self._measure_usage)

    def finish(self):
        if self.keep_measuring is False:
            return 0
        self.keep_measuring = False
        max_usage = self.monitor_thread.result()
        self.monitor_thread = None
        self.time_stamps.append(time())
        self.mem_stats.append(max_usage)
        return max_usage

    def _measure_usage(self):
        max_usage = 0
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        while self.keep_measuring:
            max_usage = max(
                max_usage,
                get_cuda_memory_used(dev),
            )
            sleep(self.interval)
        return max_usage

    def state_dict(self):
        return {
            "time_stamps": self.time_stamps,
            "mem_stats": self.mem_stats,
        }

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.state_dict(), f)

    def clear(self):
        self.mem_stats.clear()
        self.time_stamps.clear()


@OPHOOKS.register_module
class MemTracerOpHook(BaseOpHook):
    """
    Collect GPU memory usage information

    :param warmup: This parameter indicates how many iterations to truncate before profiling, defaults to 50
    :type warmup: int
    :param refreshrate: This parameter decides the frequency of write file, defaults to 10
    :type refreshrate: int
    :param data_prefix: The prefix of the stats data file, defaults to "memstats"
    :type data_prefix: string
    """

    def __init__(self, warmup: int = 50, refreshrate: int = 10, data_prefix: str = "memstats"):
        super().__init__()
        self.async_mem_monitor = AsyncMemoryMonitor()
        self._curiter = 0
        self._logger = get_dist_logger()
        self._count = 0
        self._warmup = warmup
        self._refreshrate = refreshrate
        self._data_prefix = data_prefix
        # in distributed environment
        if gpc.is_initialized(ParallelMode.GLOBAL):
            self._rank = gpc.get_global_rank()
        else:
            self._rank = 0

    def _isvalid(self, module) -> bool:
        assert isinstance(module, torch.nn.Module)
        return module.training

    def _resample(self):
        # calculate the average iteration time
        total_time = (self.async_mem_monitor.time_stamps[-1] - self.async_mem_monitor.time_stamps[0])
        avg_it_time = total_time / self.warmup
        self._logger.debug(f"total time for {self.warmup} iterations is {total_time}s")
        # adjust the sampling power
        power: int = round(-math.log(avg_it_time, 10)) + 1
        self._logger.debug(f"the power is {power}")
        self.async_mem_monitor.set_interval(power)

    @property
    def refreshrate(self) -> int:
        return self._refreshrate

    @property
    def warmup(self) -> int:
        return self._warmup

    @property
    def curiter(self) -> int:
        return self._curiter

    @property
    def valid_iter(self) -> int:
        return self.curiter - self.warmup

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        if self._isvalid(module):
            self.async_mem_monitor.finish()
            self.async_mem_monitor.start()

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        if self._isvalid(module):
            self.async_mem_monitor.finish()

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        if self._isvalid(module):
            self.async_mem_monitor.finish()
            self.async_mem_monitor.start()

    def post_bwd_exec(self, module: torch.nn.Module, input):
        if self._isvalid(module):
            self.async_mem_monitor.finish()

    def pre_iter(self):
        pass

    def post_iter(self):
        self.async_mem_monitor.finish()
        # in the warmup stage
        if self.curiter < self.warmup:
            pass
        # adjust the sampling rate
        elif self.curiter == self.warmup:
            # use adaptive sample rate
            self._resample()
        # record data to log file
        else:
            # every `refreshrate` times, refresh the file
            if self.valid_iter != 0 and self.valid_iter % self.refreshrate == 0:
                # output file info
                self._logger.info(f"dump a memory statistics as pickle to {self._data_prefix}-{self._rank}.pkl")
                self.save_results()
                self._count += 1
                self._logger.debug(f"data file has been refreshed {self._count} times")
        # finish a iteration
        self._curiter += 1

    def save_results(self):
        datafile = f"{self._data_prefix}-{self._rank}.pkl"
        self.async_mem_monitor.save(datafile)
