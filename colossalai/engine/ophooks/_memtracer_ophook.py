import torch
from . import BaseOpHook
from concurrent.futures import ThreadPoolExecutor
from colossalai.registry import OPHOOKS
from colossalai.logging import get_dist_logger
from time import sleep, time
import psutil
import pickle


def get_cuda_memory_used(device):
    """
    Get the free memory info of device.
    Notice that for CPU, this function will return 1/N of the total free memory,
    where N is the world size.
    """
    ret = torch.cuda.memory_allocated()
    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()
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


@OPHOOKS.register_module
class MemTracerOpHook(BaseOpHook):
    def __init__(self, niter=5):
        super().__init__()
        self.async_mem_monitor = AsyncMemoryMonitor()
        self._niter = niter
        self._curiter = 0
        self._logger = get_dist_logger()

    def _isvalid(self, module):
        return module.training and self._curiter < self._niter

    def niter(self):
        return self._niter

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
        assert isinstance(module, torch.nn.Module)
        if self._isvalid(module):
            self.async_mem_monitor.finish()
            self.async_mem_monitor.start()
            self._logger.debug(f'BWD PRE {module.__class__.__name__}')

    def post_bwd_exec(self, module: torch.nn.Module, input):
        assert isinstance(module, torch.nn.Module)
        if self._isvalid(module):
            self.async_mem_monitor.finish()
            self._logger.debug(f'BWD POST {module.__class__.__name__}')

    def pre_iter(self):
        pass

    def post_iter(self):
        self.async_mem_monitor.finish()
        if self._curiter == self._niter:
            self._logger.info(
                f'dump a memory statistics as pickle to ./memstats.pkl')
            self.save_results("memstats.pkl")
        self._curiter += 1

    def save_results(self, filename):
        self.async_mem_monitor.save(filename)
