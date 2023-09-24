from contextlib import contextmanager
from typing import Callable, Dict, Tuple

import torch

__all__ = [
    "_LEGACY_TENSOR_CONSTRUCTOR",
    "_NO_META_FACTORY",
    "_NORMAL_FACTORY",
    "ConstructorManager",
]

# reference: https://pytorch.org/cppdocs/notes/tensor_creation.html
_NORMAL_FACTORY = [
    "arange",
    "full",
    "empty",
    "linspace",
    "logspace",
    "ones",
    "rand",
    "randn",
    "randint",
    "randperm",
    "zeros",
    "tensor",
]

# factory function that does not support meta tensor backend
_NO_META_FACTORY = [
    "eye",
]

_LEGACY_TENSOR_CONSTRUCTOR = {
    "FloatTensor": torch.float,
    "DoubleTensor": torch.double,
    "HalfTensor": torch.half,
    "BFloat16Tensor": torch.bfloat16,
    "ByteTensor": torch.uint8,
    "CharTensor": torch.int8,
    "ShortTensor": torch.short,
    "IntTensor": torch.int,
    "LongTensor": torch.long,
    "BoolTensor": torch.bool,
}


class ConstructorManager:
    # function name: (new, old)
    overwrites: Dict[str, Tuple[Callable, Callable]] = {}
    changed: bool = False

    @staticmethod
    def apply(overwrites: Dict[Callable, Callable]):
        ConstructorManager.overwrites.clear()
        ConstructorManager.overwrites.update(overwrites)
        ConstructorManager.redo()

    @staticmethod
    def undo():
        assert ConstructorManager.changed, "No constructor change to undo"
        for name, (new, old) in ConstructorManager.overwrites.items():
            setattr(torch, name, old)
        ConstructorManager.changed = False

    @staticmethod
    def redo():
        assert not ConstructorManager.changed, "Constructor already changed"
        for name, (new, old) in ConstructorManager.overwrites.items():
            setattr(torch, name, new)
        ConstructorManager.changed = True

    @staticmethod
    @contextmanager
    def disable():
        enabled = ConstructorManager.changed
        if enabled:
            ConstructorManager.undo()
        yield
        if enabled:
            ConstructorManager.redo()

    @staticmethod
    def clear():
        if ConstructorManager.changed:
            ConstructorManager.undo()
        ConstructorManager.overwrites.clear()
