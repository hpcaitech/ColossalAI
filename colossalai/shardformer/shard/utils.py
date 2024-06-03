from typing import Set

import torch.nn as nn


def set_tensors_to_none(model: nn.Module, exclude: Set[nn.Module] = set()) -> None:
    """Set all parameters and buffers of model to None

    Args:
        model (nn.Module): The model to set
    """
    if model in exclude:
        return
    for child in model.children():
        set_tensors_to_none(child, exclude=exclude)
    for n, p in model.named_parameters(recurse=False):
        setattr(model, n, None)
    for n, buf in model.named_buffers(recurse=False):
        setattr(model, n, None)
