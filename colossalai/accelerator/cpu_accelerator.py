#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Tuple, Union

import torch

from .base_accelerator import BaseAccelerator

__all__ = ["CpuAccelerator"]


class CpuAccelerator(BaseAccelerator):
    """
    Accelerator class for cpu.
    """

    def __init__(self):
        super().__init__(name="cpu", communication_backend="gloo", is_synchronous=False)

    # =======================
    # device APIs
    # =======================
    def get_current_device(self) -> torch.device:
        """
        Return the current device.
        """
        return torch.device("cpu")

    def current_device(self) -> int:
        """
        Return the current device index.
        """

    def set_device(self, device: Union[torch.device, int]) -> None:
        """
        Bind the current process to a device.
        """

    def get_device_name(self, device: Union[torch.device, int]) -> str:
        """
        Return the name of the device.
        """

    def synchronize(self, device: Union[torch.device, int] = None):
        """
        Synchronize the current process.
        """

    def is_available(self):
        """
        Check if the accelerator is available.
        """
        return True

    def device_count(self):
        """
        Return the number of devices on the machine.
        """

    def set_to_device(self, models: Any) -> Any:
        """
        Send model to gpu.

        :param models: nn.module or a list of module
        """
        if isinstance(models, list) and len(models) > 1:
            ret = []
            for model in models:
                ret.append(model.to(self.get_current_device()))
            return ret
        elif isinstance(models, list):
            return models[0].to(self.get_current_device())
        else:
            return models.to(self.get_current_device())

    def get_device_capability(self, device=None) -> Tuple[int, int]:
        """
        Gets the cuda capability of a device.
        """

    def get_device_name(self, device=None) -> str:
        """
        Gets the name of a device.
        """

    def get_device_properties(self, device):
        """
        Gets the properties of a device.
        """

    def utilization(self, device=None) -> int:
        """
        Returns the percent of time over the past sample period during which one or more kernels was executing on the GPU as given by nvidia-smi
        """

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

    def set_rng_state(self, new_state: torch.ByteTensor, device: str = None) -> None:
        """
        Sets the random number generator state of the specified GPU.
        """
        torch.set_rng_state(new_state)

    def set_rng_state_all(self, new_states: List[torch.ByteTensor]) -> None:
        """
        Sets the random number generator state of all devices.
        """

    def manual_seed(self, seed: int) -> None:
        """
        Sets the seed for generating random numbers for the current GPU.
        """

    def manual_seed_all(self, seed: int) -> None:
        """
        Set the random seed for the all processes.
        """

    def seed(self) -> None:
        """
        Sets the seed for generating random numbers to a random number for the current GPU.
        """

    def seed_all(self) -> None:
        """
        Sets the seed for generating random numbers to a random number on all GPUs.
        """

    def initial_seed(self) -> int:
        """
        Returns the current random seed of the current GPU.
        """

    # =======================
    # memory management APIs
    # =======================

    def empty_cache(self) -> None:
        """
        Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in nvidia-smi.
        """

    def memory_stats(self, device=None) -> Dict[str, Any]:
        """
        Returns a dictionary of CUDA memory allocator statistics for a given device.
        """

    def memory_summary(self, device=None, abbreviated=False) -> str:
        """
        Returns a human-readable printout of the current memory allocator statistics for a given device.
        """

    def memory_snapshot(self):
        """
        Returns a snapshot of the CUDA memory allocator state across all devices.
        """

    def memory_allocated(self, device=None) -> int:
        """
        Returns the current GPU memory occupied by tensors in bytes for a given device.
        """

    def max_memory_allocated(self, device=None) -> int:
        """
        Returns the maximum GPU memory occupied by tensors in bytes for a given device.
        """

    def reset_max_memory_allocated(self, device=None) -> None:
        """
        Resets the starting point in tracking maximum GPU memory occupied by tensors for a given device.
        """

    def reset_max_memory_cached(self, device=None) -> None:
        """
        Resets the starting point in tracking maximum GPU memory managed by the caching allocator for a given device.
        """

    def memory_reserved(self, device=None) -> int:
        """
        Returns the current GPU memory managed by the caching allocator in bytes for a given device.
        """

    def max_memory_reserved(self, device=None) -> int:
        """
        Returns the maximum GPU memory managed by the caching allocator in bytes for a given device.
        """

    def set_per_process_memory_fraction(self, fraction: float, device=None) -> None:
        """
        Set memory fraction for a process.
        """

    def reset_peak_memory_stats(self, device=None) -> None:
        """
        Resets the "peak" stats tracked by the CUDA memory allocator.
        """

    # =======================
    # streams and events APIs
    # =======================

    def Stream(self, device=None, priority=0, **kwargs):
        """
        A CUDA stream is a linear sequence of execution that belongs to a specific device, independent from other streams. See cuda-semantics for details.
        """

    def Event(self, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False):
        """
        CUDA events are synchronization markers that can be used to monitor the device's progress, to accurately measure timing, and to synchronize CUDA streams.
        """

    def current_stream(self, device=None):
        """
        Returns the currently selected Stream for a given device.
        """

    def default_stream(self, device=None):
        """
        Returns the default Stream for a given device.
        """

    def set_stream(self, stream_):
        """
        Sets the current stream.This is a wrapper API to set the stream.
        """

    def stream(self, stream_):
        """
        Wrapper around the Context-manager StreamContext that selects a given stream.
        """

    # =======================
    # amp APIs
    # =======================
    def autocast(
        self, enabled: bool = True, dtype: torch.dtype = torch.float16, cache_enabled: bool = True
    ) -> Callable:
        """
        Return autocast function
        """
