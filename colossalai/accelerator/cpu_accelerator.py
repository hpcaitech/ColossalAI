#!/usr/bin/env python

import resource
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import psutil
import torch

from .base_accelerator import BaseAccelerator

__all__ = ["CpuAccelerator"]


class CpuAccelerator(BaseAccelerator):
    support_set_device: bool = False
    """
    Accelerator class for cpu.
    """

    def __init__(self):
        super().__init__(name="cpu", communication_backend="gloo", is_synchronous=False)

    # =======================
    # device APIs
    # =======================
    def get_version(self) -> str:
        """
        Return the version of the accelerator which torch is built against.
        """
        return ""

    def get_current_device(self) -> torch.device:
        """
        Return the current device.
        """
        return torch.device("cpu")

    def current_device(self) -> int:
        """
        Return the current device index.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def set_device(self, device: Optional[Union[torch.device, int]] = None) -> None:
        """
        Bind the current process to a device.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def get_device_name(self, device: Union[torch.device, int]) -> str:
        """
        Return the name of the device.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def synchronize(self, device: Union[torch.device, int] = None):
        """
        Synchronize the current process.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def is_available(self):
        """
        Check if the accelerator is available.
        """
        return True

    def device_count(self):
        """
        Return the number of devices on the machine.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def get_device_capability(self, device=None) -> Tuple[int, int]:
        """
        Gets the cuda capability of a device.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def get_device_name(self, device=None) -> str:
        """
        Gets the name of a device.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def get_device_properties(self, device):
        """
        Gets the properties of a device.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def utilization(self, device=None) -> int:
        """
        Returns the percent of time over the past sample period during which one or more kernels was executing on the GPU as given by nvidia-smi
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    # =======================
    # random number generator APIs
    # =======================
    def get_rng_state(self, device=None) -> torch.Tensor:
        """
        Returns the random number generator state of the specified GPU as a ByteTensor.
        """
        return torch.get_rng_state(device)

    def get_rng_state_all(self) -> List[torch.Tensor]:
        """
        Returns a list of ByteTensor representing the random number states of all devices.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def set_rng_state(self, new_state: torch.ByteTensor, device: str = None) -> None:
        """
        Sets the random number generator state of the specified GPU.
        """
        torch.set_rng_state(new_state)

    def set_rng_state_all(self, new_states: List[torch.ByteTensor]) -> None:
        """
        Sets the random number generator state of all devices.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def manual_seed(self, seed: int) -> None:
        """
        Sets the seed for generating random numbers for the current GPU.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def manual_seed_all(self, seed: int) -> None:
        """
        Set the random seed for the all processes.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def seed(self) -> None:
        """
        Sets the seed for generating random numbers to a random number for the current GPU.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def seed_all(self) -> None:
        """
        Sets the seed for generating random numbers to a random number on all GPUs.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def initial_seed(self) -> int:
        """
        Returns the current random seed of the current GPU.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    # =======================
    # memory management APIs
    # =======================

    def empty_cache(self) -> None:
        """
        Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in nvidia-smi.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def memory_stats(self, device=None) -> Dict[str, Any]:
        """
        Returns a dictionary of CUDA memory allocator statistics for a given device.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def memory_summary(self, device=None, abbreviated=False) -> str:
        """
        Returns a human-readable printout of the current memory allocator statistics for a given device.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def memory_snapshot(self):
        """
        Returns a snapshot of the CUDA memory allocator state across all devices.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def memory_allocated(self, device=None) -> int:
        """
        Returns the current GPU memory occupied by tensors in bytes for a given device.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def max_memory_allocated(self, device=None) -> int:
        """
        Returns the maximum GPU memory occupied by tensors in bytes for a given device.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def reset_max_memory_allocated(self, device=None) -> None:
        """
        Resets the starting point in tracking maximum GPU memory occupied by tensors for a given device.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def reset_max_memory_cached(self, device=None) -> None:
        """
        Resets the starting point in tracking maximum GPU memory managed by the caching allocator for a given device.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def memory_reserved(self, device=None) -> int:
        """
        Returns the current GPU memory managed by the caching allocator in bytes for a given device.
        """
        return psutil.Process().memory_info().rss

    def max_memory_reserved(self, device=None) -> int:
        """
        Returns the maximum GPU memory managed by the caching allocator in bytes for a given device.
        """
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def set_per_process_memory_fraction(self, fraction: float, device=None) -> None:
        """
        Set memory fraction for a process.
        """
        max_memory = int(psutil.virtual_memory().total * fraction)
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, hard))

    def reset_peak_memory_stats(self, device=None) -> None:
        """
        Resets the "peak" stats tracked by the CUDA memory allocator.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    # =======================
    # streams and events APIs
    # =======================

    def Stream(self, device=None, priority=0, **kwargs):
        """
        A CUDA stream is a linear sequence of execution that belongs to a specific device, independent from other streams. See cuda-semantics for details.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def Event(self, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False):
        """
        CUDA events are synchronization markers that can be used to monitor the device's progress, to accurately measure timing, and to synchronize CUDA streams.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def current_stream(self, device=None):
        """
        Returns the currently selected Stream for a given device.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def default_stream(self, device=None):
        """
        Returns the default Stream for a given device.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def set_stream(self, stream_):
        """
        Sets the current stream.This is a wrapper API to set the stream.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    def stream(self, stream_):
        """
        Wrapper around the Context-manager StreamContext that selects a given stream.
        """
        raise RuntimeError("this method is not supported for cpu accelerator")

    # =======================
    # amp APIs
    # =======================
    def autocast(
        self, enabled: bool = True, dtype: torch.dtype = torch.float16, cache_enabled: bool = True
    ) -> Callable:
        """
        Return autocast function
        """
        return nullcontext
