from typing import Optional

from torch.nn import Module

__all__ = [
    "get_pretrained_path",
    "set_pretrained_path",
]


def get_pretrained_path(model: Module) -> Optional[str]:
    return getattr(model, "_pretrained", None)


def set_pretrained_path(model: Module, path: str) -> None:
    setattr(model, "_pretrained", path)
