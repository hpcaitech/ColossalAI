from concurrent.futures import ThreadPoolExecutor
from time import sleep, time
import pickle

from colossalai.utils import get_current_device
import torch


def get_cuda_memory_used(device: torch.device) -> int:
    """
    Get the free memory info of device.
    :param device: device id
    :type device: torch.device
    :return: current memory usage, sized by MB
    :rtype: int
    """

    assert device.type == 'cuda'

    ret: int = torch.cuda.memory_allocated(device)
    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):    # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats(device)
    return ret


class AsyncMemoryMonitor:
    """
    An Async Memory Monitor runing during computing. Sampling memory usage of the current GPU
    at interval of 1/(10**power) sec.

    :param power: the power of time interval, defaults to 10
    :type power: int

    Usage:

    ```python
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
    ```
    """

    def __init__(self, power: int = 10):
        self.keep_measuring = False

        current_device = get_current_device()
        def _set_cuda_device():
            torch.cuda.set_device(current_device)

        self.executor = ThreadPoolExecutor(max_workers=1, initializer=_set_cuda_device)
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
        while self.keep_measuring:
            max_usage = max(
                max_usage,
                get_cuda_memory_used(torch.device(f'cuda:{get_current_device()}')),
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
            print(self.state_dict())
            pickle.dump(self.state_dict(), f)

    def clear(self):
        self.mem_stats.clear()
        self.time_stamps.clear()
