import torch
import gc
import psutil
from collections import namedtuple

from colossalai.context.parallel_mode import ParallelMode
from colossalai.utils import get_current_device
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
from colossalai.logging import get_dist_logger
from packaging import version

_GLOBAL_CUDA_MEM_FRACTION = 1.0
_GLOBAL_CPU_MEM_CAPACITY = -1


def _bytes_to_MB(val, decimal=2):
    """A byte-to-Megabyte converter, default using binary notation.

    :param val: X bytes to convert
    :return: X' MB
    """
    return round(val / (1024 * 1024), decimal)


# copy from PatrickStar
def _get_cpu_memory_info():
    ps_mem_info = namedtuple("ps_mem_info", ["total", "free", "cached", "buffers", "used"])
    try:
        # psutil reads the memory info from /proc/memory_info,
        # which results in returning the host memory instead of
        # that of container.
        # Here we try to read the container memory with method in:
        # https://stackoverflow.com/a/46213331/5163915
        mems = {}
        with open("/sys/fs/cgroup/memory/memory.meminfo", "rb") as f:
            for line in f:
                fields = line.split()
                mems[fields[0]] = int(fields[1]) * 1024
        total = mems[b"MemTotal:"]
        free = mems[b"MemFree:"]
        cached = mems[b"Cached:"]
        buffers = mems[b"Buffers:"]
        used = total - free - cached - buffers
        if used < 0:
            used = total - free
        mem_info = ps_mem_info(total=total, free=free, cached=cached, buffers=buffers, used=used)
    except FileNotFoundError:
        mems = psutil.virtual_memory()
        mem_info = ps_mem_info(
            total=mems.total,
            free=mems.free,
            cached=mems.cached,
            buffers=mems.buffers,
            used=mems.used,
        )
    return mem_info


def report_memory_usage(message, logger=None, report_cpu=False):
    """Calculate and print RAM usage (in GB)

    Args:
        message (str): A prefix message to add in the log.
        logger (:class:`colossalai.logging.DistributedLogger`): The logger used to record memory information.
        report_cpu (bool, optional): Whether to report CPU memory.

    Raises:
        EnvironmentError: Raise error if no distributed environment has been initialized.
    """
    if not gpc.is_initialized(ParallelMode.GLOBAL):
        raise EnvironmentError("No distributed environment is initialized")

    gpu_allocated = _bytes_to_MB(torch.cuda.memory_allocated())
    gpu_max_allocated = _bytes_to_MB(torch.cuda.max_memory_allocated())
    gpu_cached = _bytes_to_MB(torch.cuda.memory_reserved())
    gpu_max_cached = _bytes_to_MB(torch.cuda.max_memory_reserved())

    full_log = f"{message}: GPU: allocated {gpu_allocated} MB, max allocated {gpu_max_allocated} MB, " \
        + f"cached: {gpu_cached} MB, max cached: {gpu_max_cached} MB"

    if report_cpu:
        # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
        gc.collect()
        vm_stats = psutil.virtual_memory()
        vm_used = _bytes_to_MB(vm_stats.total - vm_stats.available)
        full_log += f", CPU Virtual Memory: used = {vm_used} MB, percent = {vm_stats.percent}%"

    if logger is None:
        logger = get_dist_logger()
    logger.info(full_log)

    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):    # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()


def colo_device_memory_capacity(device: torch.device) -> int:
    """
    Get the capacity of the memory of the device

    Args:
        device (torch.device): a device

    Returns:
        int: size in byte
    """
    assert isinstance(device, torch.device)
    if device.type == 'cpu':
        # In the context of 1-CPU-N-GPU, the memory capacity of the current process is 1/N overall CPU memory.
        return colo_get_cpu_memory_capacity() / gpc.num_processes_on_current_node
    if device.type == 'cuda':
        return torch.cuda.get_device_properties(get_current_device()).total_memory * _GLOBAL_CUDA_MEM_FRACTION


def colo_device_memory_used(device: torch.device) -> int:
    """
    Get the device memory on device belonging to the current process.

    Args:
        device (torch.device): a device

    Returns:
        int: memory size in bytes
    """
    if device.type == 'cpu':
        mem_info = _get_cpu_memory_info()
        # In the context of 1-CPU-N-GPU, the memory usage of the current process is 1/N CPU memory used.
        # Each process consumes the same amount of memory.
        ret = mem_info.used / gpc.num_processes_on_current_node
        return ret
    elif device.type == 'cuda':
        ret: int = torch.cuda.memory_allocated(device)
        # get the peak memory to report correct data, so reset the counter for the next call
        if hasattr(torch.cuda, "reset_peak_memory_stats"):    # pytorch 1.4+
            torch.cuda.reset_peak_memory_stats(device)
        return ret


def colo_set_process_memory_fraction(ratio: float) -> None:
    """colo_set_process_memory_fraction 

    set how much cuda memory used on the gpu belonging to the current process.

    Args:
        ratio (float): a ratio between 0. ~ 1.
    """
    if version.parse(torch.__version__) < version.parse('1.8'):
        logger = get_dist_logger('colo_set_process_memory_fraction')
        logger.warning('colo_set_process_memory_fraction failed because torch version is less than 1.8')
        return
    global _GLOBAL_CUDA_MEM_FRACTION
    _GLOBAL_CUDA_MEM_FRACTION = ratio
    torch.cuda.set_per_process_memory_fraction(_GLOBAL_CUDA_MEM_FRACTION, get_current_device())


def colo_set_cpu_memory_capacity(size: int) -> None:
    global _GLOBAL_CPU_MEM_CAPACITY
    mem_info = _get_cpu_memory_info()
    total_size = mem_info.total
    if size <= total_size:
        _GLOBAL_CPU_MEM_CAPACITY = size
    else:
        _GLOBAL_CPU_MEM_CAPACITY = total_size


def colo_get_cpu_memory_capacity() -> int:
    """
    Get the cpu memory capacity. We may not use all of it.
    Returns:
        int: _description_
    """
    global _GLOBAL_CPU_MEM_CAPACITY
    if _GLOBAL_CPU_MEM_CAPACITY == -1:
        mem_info = _get_cpu_memory_info()
        return mem_info.total
    else:
        return _GLOBAL_CPU_MEM_CAPACITY
