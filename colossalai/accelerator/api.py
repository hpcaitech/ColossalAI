#!/usr/bin/env python
from collections import OrderedDict
from typing import Union

from .base_accelerator import BaseAccelerator
from .cpu_accelerator import CpuAccelerator
from .cuda_accelerator import CudaAccelerator
from .npu_accelerator import NpuAccelerator

__all__ = ["set_accelerator", "auto_set_accelerator", "get_accelerator"]


_ACCELERATOR = None


# we use ordered dictionary here to associate the
# order with device check priority
# i.e. auto_set_accelerator will check cuda first
_ACCELERATOR_MAPPING = OrderedDict(cuda=CudaAccelerator, npu=NpuAccelerator, cpu=CpuAccelerator)


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


def auto_set_accelerator() -> None:
    """
    Automatically check if any accelerator is available.
    If an accelerator is available, set it as the global accelerator.
    """
    global _ACCELERATOR

    for accelerator_name, accelerator_cls in _ACCELERATOR_MAPPING.items():
        try:
            accelerator = accelerator_cls()
            if accelerator_name == "cpu" or accelerator.is_available():
                _ACCELERATOR = accelerator
                break
        except:
            pass

    if _ACCELERATOR is None:
        raise RuntimeError("No accelerator is available.")


def get_accelerator() -> BaseAccelerator:
    """
    Return the accelerator for the current process. If the accelerator is not initialized, it will be initialized
    to the default accelerator type.

    Returns: the accelerator for the current process.
    """
    global _ACCELERATOR

    if _ACCELERATOR is None:
        auto_set_accelerator()
    return _ACCELERATOR
