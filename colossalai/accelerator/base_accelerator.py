#!/usr/bin/env python

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

__all__ = ["BaseAccelerator"]


class BaseAccelerator(ABC):
    support_set_device: bool = True

    def __init__(self, name: str, communication_backend: str, is_synchronous: bool) -> None:
        self._name = name
        self._communication_backend = communication_backend
        self._is_synchronous = is_synchronous

    # =======================
    # immutable attributes
    # =======================

    @property
    def name(self) -> str:
        """
        Return the name of the accelerator.
        """
        return self._name

    @property
    def communication_backend(self) -> str:
        """
        Return the name of the backend communication library.
        """
        return self._communication_backend

    @property
    def is_synchronous(self) -> bool:
        """
        Return whether the accelerator is a synchronous device.
        """
        return self._is_synchronous

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name}(name={self._name}, communication_backend={self._communication_backend}, is_synchronous={self._is_synchronous})"

    # =======================
    # device APIs
    # =======================
    @abstractmethod
    def get_version(self) -> str:
        """
        Return the version of the accelerator which torch is built against.
        """

    @abstractmethod
    def get_current_device(self) -> torch.device:
        """
        Return the current device.
        """

    @abstractmethod
    def current_device(self) -> int:
        """
        Return the current device index.
        """

    @abstractmethod
    def set_device(self, device: Optional[Union[torch.device, int]] = None) -> None:
        """
        Bind the current process to a device.
        """

    @abstractmethod
    def get_device_name(self, device: Union[torch.device, int]) -> str:
        """
        Return the name of the device.
        """

    @abstractmethod
    def synchronize(self, device: Union[torch.device, int] = None):
        """
        Synchronize the current process.
        """

    @abstractmethod
    def is_available(self):
        """
        Check if the accelerator is available.
        """

    @abstractmethod
    def device_count(self):
        """
        Return the number of devices on the machine.
        """

    def set_to_device(self, models: Any) -> Any:
        """
        Send model to device.

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

    @abstractmethod
    def get_device_capability(self, device=None) -> Tuple[int, int]:
        """
        Gets the capability of a device.
        """

    @abstractmethod
    def get_device_name(self, device=None) -> str:
        """
        Gets the name of a device.
        """

    @abstractmethod
    def get_device_properties(self, device):
        """
        Gets the properties of a device.
        """

    @abstractmethod
    def utilization(self, device=None) -> int:
        """
        Returns the percent of time over the past sample period during which one or more kernels was executing on the device as given by nvidia-smi or npu-smi, etc.
        """

    # =======================
    # random number generator APIs
    # =======================
    @abstractmethod
    def get_rng_state(self, device="cuda") -> torch.Tensor:
        """
        Returns the random number generator state of the specified device as a ByteTensor.
        """

    @abstractmethod
    def get_rng_state_all(self) -> List[torch.Tensor]:
        """
        Returns a list of ByteTensor representing the random number states of all devices.
        """

    @abstractmethod
    def set_rng_state(self, new_state: torch.ByteTensor, device: str = "cuda") -> None:
        """
        Sets the random number generator state of the specified device.
        """

    @abstractmethod
    def set_rng_state_all(self, new_states: List[torch.ByteTensor]) -> None:
        """
        Sets the random number generator state of all devices.
        """

    @abstractmethod
    def manual_seed(self, seed: int) -> None:
        """
        Sets the seed for generating random numbers for the current device.
        """

    @abstractmethod
    def manual_seed_all(self, seed: int) -> None:
        """
        Sets the seed for generating random numbers on all devices.
        """

    @abstractmethod
    def seed(self) -> None:
        """
        Sets the seed for generating random numbers to a random number for the current device.
        """

    @abstractmethod
    def seed_all(self) -> None:
        """
        Sets the seed for generating random numbers to a random number on all devices.
        """

    @abstractmethod
    def initial_seed(self) -> int:
        """
        Returns the current random seed of the current device.
        """

    # =======================
    # memory management APIs
    # =======================
    @abstractmethod
    def empty_cache(self) -> None:
        """
        Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other device application and visible in nvidia-smi.
        """

    @abstractmethod
    def memory_stats(self, device=None) -> Dict[str, Any]:
        """
        Returns a dictionary of CUDA memory allocator statistics for a given device.
        """

    @abstractmethod
    def memory_summary(self, device=None, abbreviated=False) -> str:
        """
        Returns a human-readable printout of the current memory allocator statistics for a given device.
        """

    @abstractmethod
    def memory_snapshot(self):
        """
        Returns a snapshot of the CUDA memory allocator state across all devices.
        """

    @abstractmethod
    def memory_allocated(self, device=None) -> int:
        """
        Returns the current device memory occupied by tensors in bytes for a given device.
        """

    @abstractmethod
    def max_memory_allocated(self, device=None) -> int:
        """
        Returns the maximum device memory occupied by tensors in bytes for a given device.
        """

    @abstractmethod
    def reset_max_memory_allocated(self, device=None) -> None:
        """
        Resets the starting point in tracking maximum device memory occupied by tensors for a given device.
        """

    @abstractmethod
    def reset_max_memory_cached(self, device=None) -> None:
        """
        Resets the starting point in tracking maximum device memory managed by the caching allocator for a given device.
        """

    @abstractmethod
    def memory_reserved(self, device=None) -> int:
        """
        Returns the current device memory managed by the caching allocator in bytes for a given device.
        """

    @abstractmethod
    def max_memory_reserved(self, device=None) -> int:
        """
        Returns the maximum device memory managed by the caching allocator in bytes for a given device.
        """

    @abstractmethod
    def set_per_process_memory_fraction(self, fraction: float, device=None) -> None:
        """
        Set memory fraction for a process.
        """

    @abstractmethod
    def reset_peak_memory_stats(self, device=None) -> None:
        """
        Resets the "peak" stats tracked by the device memory allocator.
        """

    # =======================
    # streams and events APIs
    # =======================

    @abstractmethod
    def Stream(self, device=None, priority=0, **kwargs):
        """
        A device stream is a linear sequence of execution that belongs to a specific device, independent from other streams. See cuda-semantics for details.
        """

    @abstractmethod
    def Event(self, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False):
        """
        device events are synchronization markers that can be used to monitor the device's progress, to accurately measure timing, and to synchronize CUDA streams.
        """

    @abstractmethod
    def current_stream(self, device=None):
        """
        Returns the currently selected Stream for a given device.
        """

    @abstractmethod
    def default_stream(self, device=None):
        """
        Returns the default Stream for a given device.
        """

    @abstractmethod
    def set_stream(self, stream_):
        """
        Sets the current stream.This is a wrapper API to set the stream.
        """

    @abstractmethod
    def stream(self, stream_):
        """
        Wrapper around the Context-manager StreamContext that selects a given stream.
        """

    # =======================
    # amp APIs
    # =======================
    @abstractmethod
    def autocast(
        self, enabled: bool = True, dtype: torch.dtype = torch.float16, cache_enabled: bool = True
    ) -> Callable:
        """
        Return autocast function
        """
