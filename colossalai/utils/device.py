#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Any, Dict, List, Optional, Tuple, Callable

import torch
import torch.distributed as dist

IS_NPU_AVAILABLE: bool = False
try:
    import torch_npu  # noqa

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


def _dispatch_device_func(fn_name: str, *args, **kwargs):
    if torch.cuda.is_available():
        return getattr(torch.cuda, fn_name)(*args, **kwargs)
    elif IS_NPU_AVAILABLE:
        return getattr(torch.npu, fn_name)(*args, **kwargs)
    else:
        raise RuntimeError("No device available")


# device semantics


def can_device_access_peer(device, peer_device) -> bool:
    return _dispatch_device_func("can_device_access_peer", device, peer_device)


def current_device() -> int:
    return _dispatch_device_func("current_device")


def current_stream(device=None):
    return _dispatch_device_func("current_stream", device)


def default_stream(device=None):
    return _dispatch_device_func("default_stream", device)


def device_count() -> int:
    return _dispatch_device_func("device_count")


def get_device_capability(device=None) -> Tuple[int, int]:
    return _dispatch_device_func("get_device_capability", device)


def get_device_name(device=None) -> str:
    return _dispatch_device_func("get_device_name", device)


def get_device_properties(device):
    return _dispatch_device_func("get_device_properties", device)


def set_device(index: Optional[int] = None) -> None:
    if index is None:
        index = dist.get_rank() % device_count()
    _dispatch_device_func("set_device", index)


def set_stream(stream_):
    return _dispatch_device_func("set_stream", stream_)


def stream(stream_):
    return _dispatch_device_func("stream", stream_)


def synchronize():
    return _dispatch_device_func("synchronize")


def utilization(device=None) -> int:
    return _dispatch_device_func("utilization", device)


# random number generator


def get_rng_state(device="cuda") -> torch.Tensor:
    return _dispatch_device_func("get_rng_state", device)


def get_rng_state_all() -> List[torch.Tensor]:
    return _dispatch_device_func("get_rng_state_all")


def set_rng_state(new_state: torch.ByteTensor, device="cuda") -> None:
    return _dispatch_device_func("set_rng_state", new_state, device)


def set_rng_state_all(new_states: List[torch.ByteTensor]) -> None:
    return _dispatch_device_func("set_rng_state_all", new_states)


def manual_seed(seed: int) -> None:
    return _dispatch_device_func("manual_seed", seed)


def manual_seed_all(seed: int) -> None:
    return _dispatch_device_func("manual_seed_all", seed)


def seed() -> None:
    return _dispatch_device_func("seed")


def seed_all() -> None:
    return _dispatch_device_func("seed_all")


def initial_seed() -> int:
    return _dispatch_device_func("initial_seed")


# streams and events


def Stream(device=None, priority=0, **kwargs):
    return _dispatch_device_func("Stream", device, priority, **kwargs)


def Event(enable_timing: bool = False, blocking: bool = False, interprocess: bool = False):
    return _dispatch_device_func("Event", enable_timing, blocking, interprocess)


# memory management


def empty_cache() -> None:
    return _dispatch_device_func("empty_cache")


def memory_stats(device=None) -> Dict[str, Any]:
    return _dispatch_device_func("memory_stats", device)


def memory_summary(device=None, abbreviated=False) -> str:
    return _dispatch_device_func("memory_summary", device, abbreviated)


def memory_snapshot():
    return _dispatch_device_func("memory_snapshot")


def memory_allocated(device=None) -> int:
    return _dispatch_device_func("memory_allocated", device)


def max_memory_allocated(device=None) -> int:
    return _dispatch_device_func("max_memory_allocated", device)


def reset_max_memory_allocated(device=None) -> None:
    return _dispatch_device_func("reset_max_memory_allocated", device)


def reset_max_memory_cached(device=None) -> None:
    return _dispatch_device_func("reset_max_memory_cached", device)


def memory_reserved(device=None) -> int:
    return _dispatch_device_func("memory_reserved", device)


def max_memory_reserved(device=None) -> int:
    return _dispatch_device_func("max_memory_reserved", device)


def set_per_process_memory_fraction(fraction: float, device=None) -> None:
    return _dispatch_device_func("set_per_process_memory_fraction", fraction, device)


def reset_peak_memory_stats(device=None) -> None:
    return _dispatch_device_func("reset_peak_memory_stats", device)


# amp


def autocast() -> Callable:
    if torch.cuda.is_available():
        return torch.cuda.amp.autocast()
    elif IS_NPU_AVAILABLE:
        return torch.npu.amp.autocast()
    else:
        raise RuntimeError("No device available")
