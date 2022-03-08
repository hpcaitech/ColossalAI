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


def get_cuda_memory_used(device: Optional[torch.device]) -> int:
    """
    Get the free memory info of device.
    Notice that for CPU, this function will return 1/N of the total free memory,
    where N is the world size.
    """
    ret: int = torch.cuda.memory_allocated(device)
    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):    # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats(device)
    return ret


class AsyncMemoryMonitor:

    def __init__(self, power=10):
        """
        An Async Mem Monitor runing during computing.
        Sampling GPU memory usage of the current GPU dev
        at interval of 1/(10**power) sec.
        """
        self.keep_measuring = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.monitor_thread = None
        self.interval = 1 / (10**power)
        self.time_stamps = []
        self.mem_stats = []

    def __len__(self):
        return len(self.mem_stats)

    def set_interval(self, power: int):
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
    '''
    Collect GPU memory usage information

    Args:
        warmup (int): This parameter indicates how many iterations to truncate
        before profiling, e.g. set to 5 and the data will start from 6-th iteration
        refreshrate (int): This parameter decides the frequency of write file.
        datafile(string): the name of the stats data file
    Attributes:
        _warmup (int): warmup iterations
        _refreshrate(int): how many iterations we shall refresh the file
        _logger (colossalai.logging.logger): output log file
        _curiter (int): current iteration number
        _count (int): the number of times the data file was written
        _data_prefix (string): the prefix of the stats data file
        _rank (int): the rank of current node
    '''

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
            self._logger.debug(f'FWD PRE {module.__class__.__name__}')

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        if self._isvalid(module):
            self.async_mem_monitor.finish()
            self._logger.debug(f'FWD POST {module.__class__.__name__}')

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        if self._isvalid(module):
            self.async_mem_monitor.finish()
            self.async_mem_monitor.start()
            self._logger.debug(f'BWD PRE {module.__class__.__name__}')

    def post_bwd_exec(self, module: torch.nn.Module, input):
        if self._isvalid(module):
            self.async_mem_monitor.finish()
            self._logger.debug(f'BWD POST {module.__class__.__name__}')

    def pre_iter(self):
        pass

    def post_iter(self):
        self.async_mem_monitor.finish()
        # in the warmup stage
        if self._curiter < self.warmup:
            # TODO: record time and adaptively change sampling rate
            pass
        elif self._curiter == self._warmup:
            self.async_mem_monitor.clear()
        else:
            # every `refreshrate` times, refresh the file
            if self.valid_iter != 0 and self.valid_iter % self.refreshrate == 0:
                # output file info
                self._logger.info(f'dump a memory statistics as pickle to {self._dataprefix}-{self._rank}.pkl')
                self.save_results()
                self._count += 1
                self._logger.debug(f'data file has been refreshed {self._count} times')
        # finish a iteration
        self._curiter += 1

    def save_results(self):
        datafile = f"{self._data_prefix}-{self._rank}.pkl"
        self.async_mem_monitor.save(datafile)
