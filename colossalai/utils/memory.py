from collections import namedtuple

import psutil
import torch
import torch.distributed as dist

from colossalai.utils import get_current_device

_GLOBAL_CUDA_MEM_FRACTION = 1.0
_GLOBAL_CPU_MEM_CAPACITY = -1


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


def colo_device_memory_capacity(device: torch.device) -> int:
    """
    Get the capacity of the memory of the device

    Args:
        device (torch.device): a device

    Returns:
        int: size in byte
    """
    # TODO: add NPU support
    assert isinstance(device, torch.device)
    if device.type == "cpu":
        # In the context of 1-CPU-N-GPU, the memory capacity of the current process is 1/N overall CPU memory.
        return colo_get_cpu_memory_capacity() // dist.get_world_size()
    if device.type == "cuda":
        return torch.cuda.get_device_properties(get_current_device()).total_memory * _GLOBAL_CUDA_MEM_FRACTION


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
