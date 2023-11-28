#!/usr/bin/env python
from typing import Union

from .base_accelerator import BaseAccelerator
from .cuda_accelerator import CudaAccelerator
from .npu_accelerator import NpuAccelerator

__all__ = ["set_accelerator", "get_accelerator"]


_ACCELERATOR = None

_ACCELERATOR_MAPPING = {
    "cuda": CudaAccelerator,
    "npu": NpuAccelerator,
}

_DEFAULT_ACCELERATOR_TYPE = CudaAccelerator


def set_accelerator(accelerator: Union[str, BaseAccelerator]) -> None:
    """
    Set the global accelerator for the current process.

    Args:
        accelerator (Union[str, BaseAccelerator]): the type of accelerator to which the current device belongs.
    """

    global _ACCELERATOR

    if isinstance(accelerator, str):
        _ACCELERATOR = _ACCELERATOR_MAPPING[accelerator]()
    elif isinstance(accelerator, BaseAccelerator):
        _ACCELERATOR = accelerator
    else:
        raise TypeError("accelerator must be either a string or an instance of BaseAccelerator")


def get_accelerator() -> BaseAccelerator:
    """
    Return the accelerator for the current process. If the accelerator is not initialized, it will be initialized
    to the default accelerator type.

    Returns: the accelerator for the current process.
    """
    global _ACCELERATOR

    if _ACCELERATOR is None:
        _ACCELERATOR = _DEFAULT_ACCELERATOR_TYPE()
    return _ACCELERATOR
