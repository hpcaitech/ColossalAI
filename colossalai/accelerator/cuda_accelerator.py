#!/usr/bin/env python
from typing import Union

import torch

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
    def current_device(self) -> int:
        """
        Return the current device index.
        """
        return torch.cuda.current_device()

    def set_device(self, device: Union[torch.device, int]) -> None:
        """
        Bind the current process to a device.
        """
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
