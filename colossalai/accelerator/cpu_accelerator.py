#!/usr/bin/env python


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
