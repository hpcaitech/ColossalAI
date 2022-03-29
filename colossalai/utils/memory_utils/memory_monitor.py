#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import gc

import psutil
import torch

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils.cuda import get_current_device
from typing import Optional


def colo_cuda_memory_used(device: Optional[torch.device] = None) -> int:
    """
    Get the free memory info of device.
    :param device: a torch device instance or None
    :type device: Optional[torch.device]
    :return: current memory usage, sized by Byte
    :rtype: int
    """
    if device:
        assert device.type == 'cuda'
    else:
        device = torch.device(f'cuda:{get_current_device()}')

    ret: int = torch.cuda.memory_allocated(device)
    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):    # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats(device)
    return ret


def bytes_to_GB(val, decimal=2):
    """A byte-to-Gigabyte converter, defaultly using binary notation.

    :param val: X bytes to convert
    :return: X' GB
    """
    return round(val / (1024 * 1024 * 1024), decimal)


def bytes_to_MB(val, decimal=2):
    """A byte-to-Megabyte converter, defaultly using binary notation.

    :param val: X bytes to convert
    :return: X' MB
    """
    return round(val / (1024 * 1024), decimal)


def report_memory_usage(message, logger=None, report_cpu=False):
    """Calculate and print RAM usage (in GB)

    :param message: A prefix message to add in the log
    :type message: str
    :param logger: An instance of :class:`colossalai.logging.DistributedLogger`
    :type logger: :class:`colossalai.logging.DistributedLogger`, optional
    :param report_cpu: Whether to report CPU memory
    :type report_cpu: bool, optional
    :raises EnvironmentError: Raise error if no distributed environment has been initialized
    """
    if not gpc.is_initialized(ParallelMode.GLOBAL):
        raise EnvironmentError("No distributed environment is initialized")

    gpu_allocated = bytes_to_MB(torch.cuda.memory_allocated())
    gpu_max_allocated = bytes_to_MB(torch.cuda.max_memory_allocated())
    gpu_cached = bytes_to_MB(torch.cuda.memory_reserved())
    gpu_max_cached = bytes_to_MB(torch.cuda.max_memory_reserved())

    full_log = f"{message}: GPU: allocated {gpu_allocated} MB, max allocated {gpu_max_allocated} MB, " \
        + f"cached: {gpu_cached} MB, max cached: {gpu_max_cached} MB"

    if report_cpu:
        # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
        gc.collect()
        vm_stats = psutil.virtual_memory()
        vm_used = bytes_to_MB(vm_stats.total - vm_stats.available)
        full_log += f", CPU Virtual Memory: used = {vm_used} MB, percent = {vm_stats.percent}%"

    if logger is None:
        logger = get_dist_logger()
    logger.info(full_log)

    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):    # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()
