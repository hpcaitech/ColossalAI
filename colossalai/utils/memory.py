#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import gc

import psutil
import torch

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_global_dist_logger


def bytes_to_GB(val, decimal=2):
    '''A byte-to-Gigabyte converter, defaultly using binary notation.

    :param val: X bytes to convert 
    :return: X' Gb
    '''
    return round(val / (1024 * 1024 * 1024), decimal)


def report_memory_usage(message):
    '''Calculate and print RAM usage (in GB)

    :param message: a prefix message to add in the log
    :type message: str
    :raises EnvironmentError: raise error if no distributed environment has been initialized
    '''
    if not gpc.is_initialized(ParallelMode.GLOBAL):
        raise EnvironmentError("No distributed environment is initialized")

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()
    vm_stats = psutil.virtual_memory()
    vm_used = bytes_to_GB(vm_stats.total - vm_stats.available)

    gpu_allocated = bytes_to_GB(torch.cuda.memory_allocated())
    gpu_max_allocated = bytes_to_GB(torch.cuda.max_memory_allocated())
    gpu_cached = bytes_to_GB(torch.cuda.memory_cached())
    gpu_max_cached = bytes_to_GB(torch.cuda.max_memory_cached())

    get_global_dist_logger().info(
        f"{message} - GPU: allocated {gpu_allocated}GB, max allocated {gpu_max_allocated}GB, cached: {gpu_cached} GB, "
        f"max cached: {gpu_max_cached}GB, CPU Virtual Memory: used = {vm_used}GB, percent = {vm_stats.percent}%")

    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()
