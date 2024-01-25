#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from .base_accelerator import BaseAccelerator

__all__ = ["CudaAccelerator"]


class CudaAccelerator(BaseAccelerator):
    """
    Accelerator class for Nvidia CUDA devices.
    """

    def __init__(self):
        super().__init__(name="cuda", communication_backend="nccl", is_synchronous=False)

    # =======================
    # device APIs
    # =======================
    def get_version(self) -> str:
        """
        Return the version of the accelerator which torch is built against.
        """
        return torch.version.cuda

    def get_current_device(self) -> torch.device:
        """
        Return the current device.
        """
        return torch.device(f"cuda:{torch.cuda.current_device()}")

    def current_device(self) -> int:
        """
        Return the current device index.
        """
        return torch.cuda.current_device()

    def set_device(self, device: Optional[Union[torch.device, int]] = None) -> None:
        """
        Bind the current process to a device.
        """
        if device is None:
            if not dist.is_initialized():
                raise RuntimeError("Cannot get current device when distributed is not initialized.")
            device = dist.get_rank() % self.device_count()
        torch.cuda.set_device(device)

    def get_device_name(self, device: Union[torch.device, int]) -> str:
        """
        Return the name of the device.
        """
        return torch.cuda.get_device_name(device)

    def synchronize(self, device: Union[torch.device, int] = None):
        """
        Synchronize the current process.
        """
        torch.cuda.synchronize(device)

    def is_available(self):
        """
        Check if the accelerator is available.
        """
        return torch.cuda.is_available()

    def device_count(self):
        """
        Return the number of devices on the machine.
        """
        return torch.cuda.device_count()

    def get_device_capability(self, device=None) -> Tuple[int, int]:
        """
        Gets the cuda capability of a device.
        """
        return torch.cuda.get_device_capability(device)

    def get_device_name(self, device=None) -> str:
        """
        Gets the name of a device.
        """
        return torch.cuda.get_device_name(device)

    def get_device_properties(self, device):
        """
        Gets the properties of a device.
        """
        return torch.cuda.get_device_properties(device)

    def utilization(self, device=None) -> int:
        """
        Returns the percent of time over the past sample period during which one or more kernels was executing on the GPU as given by nvidia-smi
        """
        return torch.cuda.utilization(device)

    # =======================
    # random number generator APIs
    # =======================
    def get_rng_state(self, device="cuda") -> torch.Tensor:
        """
        Returns the random number generator state of the specified GPU as a ByteTensor.
        """
        return torch.cuda.get_rng_state(device)

    def get_rng_state_all(self) -> List[torch.Tensor]:
        """
        Returns a list of ByteTensor representing the random number states of all devices.
        """
        return torch.cuda.get_rng_state_all()

    def set_rng_state(self, new_state: torch.ByteTensor, device: str = "cuda") -> None:
        """
        Sets the random number generator state of the specified GPU.
        """
        torch.cuda.set_rng_state(new_state, device)

    def set_rng_state_all(self, new_states: List[torch.ByteTensor]) -> None:
        """
        Sets the random number generator state of all devices.
        """
        torch.cuda.set_rng_state_all(new_states)

    def manual_seed(self, seed: int) -> None:
        """
        Sets the seed for generating random numbers for the current GPU.
        """
        torch.cuda.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        """
        Set the random seed for the all processes.
        """
        torch.cuda.manual_seed_all(seed)

    def seed(self) -> None:
        """
        Sets the seed for generating random numbers to a random number for the current GPU.
        """
        torch.cuda.seed()

    def seed_all(self) -> None:
        """
        Sets the seed for generating random numbers to a random number on all GPUs.
        """
        torch.cuda.seed_all()

    def initial_seed(self) -> int:
        """
        Returns the current random seed of the current GPU.
        """
        return torch.cuda.initial_seed()

    # =======================
    # memory management APIs
    # =======================

    def empty_cache(self) -> None:
        """
        Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in nvidia-smi.
        """
        torch.cuda.empty_cache()

    def memory_stats(self, device=None) -> Dict[str, Any]:
        """
        Returns a dictionary of CUDA memory allocator statistics for a given device.
        """
        return torch.cuda.memory_stats(device=device)

    def memory_summary(self, device=None, abbreviated=False) -> str:
        """
        Returns a human-readable printout of the current memory allocator statistics for a given device.
        """
        return torch.cuda.memory_summary(device=device, abbreviated=abbreviated)

    def memory_snapshot(self):
        """
        Returns a snapshot of the CUDA memory allocator state across all devices.
        """
        return torch.cuda.memory_snapshot()

    def memory_allocated(self, device=None) -> int:
        """
        Returns the current GPU memory occupied by tensors in bytes for a given device.
        """
        return torch.cuda.memory_allocated(device=device)

    def max_memory_allocated(self, device=None) -> int:
        """
        Returns the maximum GPU memory occupied by tensors in bytes for a given device.
        """
        return torch.cuda.max_memory_allocated(device=device)

    def reset_max_memory_allocated(self, device=None) -> None:
        """
        Resets the starting point in tracking maximum GPU memory occupied by tensors for a given device.
        """
        torch.cuda.reset_max_memory_allocated(device=device)

    def reset_max_memory_cached(self, device=None) -> None:
        """
        Resets the starting point in tracking maximum GPU memory managed by the caching allocator for a given device.
        """
        torch.cuda.reset_max_memory_cached(device=device)

    def memory_reserved(self, device=None) -> int:
        """
        Returns the current GPU memory managed by the caching allocator in bytes for a given device.
        """
        return torch.cuda.memory_reserved(device=device)

    def max_memory_reserved(self, device=None) -> int:
        """
        Returns the maximum GPU memory managed by the caching allocator in bytes for a given device.
        """
        return torch.cuda.max_memory_reserved(device=device)

    def set_per_process_memory_fraction(self, fraction: float, device=None) -> None:
        """
        Set memory fraction for a process.
        """
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)

    def reset_peak_memory_stats(self, device=None) -> None:
        """
        Resets the "peak" stats tracked by the CUDA memory allocator.
        """
        torch.cuda.reset_peak_memory_stats(device=device)

    # =======================
    # streams and events APIs
    # =======================

    def Stream(self, device=None, priority=0, **kwargs):
        """
        A CUDA stream is a linear sequence of execution that belongs to a specific device, independent from other streams. See cuda-semantics for details.
        """
        return torch.cuda.Stream(device, priority, **kwargs)

    def Event(self, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False):
        """
        CUDA events are synchronization markers that can be used to monitor the device's progress, to accurately measure timing, and to synchronize CUDA streams.
        """
        return torch.cuda.Event(enable_timing, blocking, interprocess)

    def current_stream(self, device=None):
        """
        Returns the currently selected Stream for a given device.
        """
        return torch.cuda.current_stream(device)

    def default_stream(self, device=None):
        """
        Returns the default Stream for a given device.
        """
        return torch.cuda.default_stream(device)

    def set_stream(self, stream_):
        """
        Sets the current stream.This is a wrapper API to set the stream.
        """
        torch.cuda.set_stream(stream_)

    def stream(self, stream_):
        """
        Wrapper around the Context-manager StreamContext that selects a given stream.
        """
        return torch.cuda.stream(stream_)

    # =======================
    # amp APIs
    # =======================
    def autocast(
        self, enabled: bool = True, dtype: torch.dtype = torch.float16, cache_enabled: bool = True
    ) -> Callable:
        """
        Return autocast function
        """
        return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)
