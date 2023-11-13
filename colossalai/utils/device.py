#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional

import torch
import torch.distributed as dist

IS_NPU_AVAILABLE: bool = False
try:
    IS_NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    pass


def set_to_cuda(models):
    """Send model to gpu.

    :param models: nn.module or a list of module
    """
    if isinstance(models, list) and len(models) > 1:
        ret = []
        for model in models:
            ret.append(model.to(get_current_device()))
        return ret
    elif isinstance(models, list):
        return models[0].to(get_current_device())
    else:
        return models.to(get_current_device())


def get_current_device() -> torch.device:
    """
    Returns currently selected device (gpu/cpu).
    If cuda available, return gpu, otherwise return cpu.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    elif IS_NPU_AVAILABLE:
        return torch.device(f"npu:{torch.npu.current_device()}")
    else:
        return torch.device("cpu")


def synchronize():
    """Similar to cuda.synchronize().
    Waits for all kernels in all streams on a CUDA device to complete.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif IS_NPU_AVAILABLE:
        torch.npu.synchronize()


def device_count() -> int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    elif IS_NPU_AVAILABLE:
        return torch.npu.device_count()
    else:
        raise RuntimeError("No device available")


def empty_cache():
    """Similar to cuda.empty_cache()
    Releases all unoccupied cached memory currently held by the caching allocator.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif IS_NPU_AVAILABLE:
        torch.npu.empty_cache()


def set_device(index: Optional[int] = None) -> None:
    if index is None:
        index = dist.get_rank() % device_count()
    if torch.cuda.is_available():
        torch.cuda.set_device(index)
    elif IS_NPU_AVAILABLE:
        torch.npu.set_device(index)


def Stream(device=None, priority=0, **kwargs):
    if torch.cuda.is_available():
        return torch.cuda.Stream(device, priority, **kwargs)
    elif IS_NPU_AVAILABLE:
        return torch.npu.Stream(device, priority, **kwargs)
    else:
        raise RuntimeError("No device available")


def stream(stream_):
    if torch.cuda.is_available():
        return torch.cuda.stream(stream_)
    elif IS_NPU_AVAILABLE:
        return torch.npu.stream(stream_)
    else:
        raise RuntimeError("No device available")


def set_stream(stream_):
    if torch.cuda.is_available():
        return torch.cuda.set_stream(stream_)
    elif IS_NPU_AVAILABLE:
        return torch.npu.set_stream(stream_)
    else:
        raise RuntimeError("No device available")


def current_stream(device=None):
    if torch.cuda.is_available():
        return torch.cuda.current_stream(device)
    elif IS_NPU_AVAILABLE:
        return torch.npu.current_stream(device)
    else:
        raise RuntimeError("No device available")


def default_stream(device=None):
    if torch.cuda.is_available():
        return torch.cuda.default_stream(device)
    elif IS_NPU_AVAILABLE:
        return torch.npu.default_stream(device)
    else:
        raise RuntimeError("No device available")
