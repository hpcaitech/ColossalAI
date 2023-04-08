import json
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from time import sleep, time

import torch

from colossalai.utils import colo_device_memory_used, get_current_device


class MemoryMonitor:
    """Base class for all types of memory monitor.
    All monitors should have a list called `time_stamps` and a list called `mem_stats`.
    """

    def __init__(self):
        self.time_stamps = []
        self.mem_stats = []

    def __len__(self):
        return len(self.mem_stats)

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def finish(self):
        pass

    def state_dict(self):
        return {
            "time_stamps": self.time_stamps,
            "mem_stats": self.mem_stats,
        }

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.state_dict(), f)

    def clear(self):
        self.mem_stats.clear()
        self.time_stamps.clear()


class AsyncMemoryMonitor(MemoryMonitor):
    """
    An Async Memory Monitor runing during computing. Sampling memory usage of the current GPU
    at interval of `1/(10**power)` sec.

    The idea comes from Runtime Memory Tracer of PatrickStar
    `PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management`_

    Usage::

        async_mem_monitor = AsyncMemoryMonitor()
        input = torch.randn(2, 20).cuda()
        OP1 = torch.nn.Linear(20, 30).cuda()
        OP2 = torch.nn.Linear(30, 40).cuda()

        async_mem_monitor.start()
        output = OP1(input)
        async_mem_monitor.finish()
        async_mem_monitor.start()
        output = OP2(output)
        async_mem_monitor.finish()
        async_mem_monitor.save('log.pkl')

    Args:
        power (int, optional): the power of time interva. Defaults to 10.

    .. _PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management:
        https://arxiv.org/abs/2108.05818
    """

    def __init__(self, power: int = 10):
        super().__init__()
        self.keep_measuring = False

        current_device = get_current_device()

        def _set_cuda_device():
            torch.cuda.set_device(current_device)

        self.executor = ThreadPoolExecutor(max_workers=1, initializer=_set_cuda_device)
        self.monitor_thread = None
        self.interval = 1 / (10**power)

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
        while self.keep_measuring:
            max_usage = max(
                max_usage,
                colo_device_memory_used(get_current_device()),
            )
            sleep(self.interval)
        return max_usage


class SyncCudaMemoryMonitor(MemoryMonitor):
    """
    A synchronized cuda memory monitor.
    It only record the maximum allocated cuda memory from start point to finish point.
    """

    def __init__(self, power: int = 10):
        super().__init__()

    def start(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def finish(self) -> int:
        """
        return max gpu memory used since latest `start()`.

        Returns:
            int: max GPU memory
        """
        torch.cuda.synchronize()
        self.time_stamps.append(time())
        max_usage = torch.cuda.max_memory_allocated()
        self.mem_stats.append(max_usage)
        return max_usage
