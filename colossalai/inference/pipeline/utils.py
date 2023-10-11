from typing import Set

import torch.nn as nn

from colossalai.shardformer._utils import getattr_, setattr_


def set_tensors_to_none(model: nn.Module, include: Set[str] = set()) -> None:
    """
    Set all parameters and buffers of model to None

    Args:
        model (nn.Module): The model to set
    """
    for module_suffix in include:
        set_module = getattr_(model, module_suffix)
        for n, p in set_module.named_parameters():
            setattr_(set_module, n, None)
        for n, buf in set_module.named_buffers():
            setattr_(set_module, n, None)
        setattr_(model, module_suffix, None)


def get_suffix_name(suffix: str, name: str):
    """
    Get the suffix name of the module, as `suffix.name` when name is string or `suffix[name]` when name is a digit,
    and 'name' when `suffix` is empty.

    Args:
        suffix (str): The suffix of the suffix module
        name (str): The name of the current module
    """
    point = "" if suffix is "" else "."
    suffix_name = suffix + f"[{name}]" if name.isdigit() else suffix + f"{point}{name}"
    return suffix_name
