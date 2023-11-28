#!/usr/bin/env python
from abc import ABC, abstractmethod
from typing import Union

import torch

__all__ = ["BaseAccelerator"]


class BaseAccelerator(ABC):
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
    def current_device(self) -> int:
        """
        Return the current device index.
        """

    @abstractmethod
    def set_device(self, device: Union[torch.device, int]) -> None:
        """
        Bind the current process to a device.
        """

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
