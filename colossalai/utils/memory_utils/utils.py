import torch
from colossalai.context.parallel_mode import ParallelMode
from colossalai.utils import get_current_device

from collections import namedtuple
import psutil
from colossalai.core import global_context as gpc

_GLOBAL_CUDA_MEM_FRACTION = 1.0


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


def colo_device_memory_used(device) -> int:
    if not isinstance(device, torch.device):
        device = torch.device(f"cuda:{device}")
    if device.type == 'cpu':
        mem_info = _get_cpu_memory_info()
        # FIXME(jiaruifang) only work for 1-CPU multi-GPU
        # CPU memory is sharded with all processes
        # Not support multi-GPU multi-CPU
        # We need a local_world_size here
        ret = mem_info.used / gpc.get_world_size(ParallelMode.DATA)
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
    global _GLOBAL_CUDA_MEM_FRACTION
    _GLOBAL_CUDA_MEM_FRACTION = ratio
    torch.cuda.set_per_process_memory_fraction(_GLOBAL_CUDA_MEM_FRACTION, get_current_device())


def colo_cuda_memory_capacity() -> float:
    """
    Get cuda memory capacity of the current cuda.
    """
    return torch.cuda.get_device_properties(get_current_device()).total_memory * _GLOBAL_CUDA_MEM_FRACTION
