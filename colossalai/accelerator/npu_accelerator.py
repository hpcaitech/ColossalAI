#!/usr/bin/env python

from typing import Union

import torch

from .base_accelerator import BaseAccelerator

try:
    import torch_npu  # noqa
except ImportError:
    pass


__all__ = ["NpuAccelerator"]


class NpuAccelerator(BaseAccelerator):
    """
    Accelerator class for Huawei NPU devices.
    """

    def __init__(self):
        super().__init__(name="npu", communication_backend="hccl", is_synchronous=False)

    # =======================
    # device APIs
    # =======================
    def current_device(self) -> int:
        """
        Return the current device index.
        """
        return torch.npu.current_device()

    def set_device(self, device: Union[torch.device, int]) -> None:
        """
        Bind the current process to a device.
        """
        torch.npu.set_device(device)

    def get_device_name(self, device: Union[torch.device, int]) -> str:
        """
        Return the name of the device.
        """
        return torch.npu.get_device_name(device)

    def synchronize(self, device: Union[torch.device, int] = None):
        """
        Synchronize the current process.
        """
        torch.npu.synchronize(device)

    def is_available(self):
        """
        Check if the accelerator is available.
        """
        return torch.npu.is_available()

    def device_count(self):
        """
        Return the number of devices on the machine.
        """
        return torch.npu.device_count()
