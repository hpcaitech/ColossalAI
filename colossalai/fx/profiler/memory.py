import torch
from torch.fx import Node
from typing import Union, Dict, List, Tuple
from . import META_COMPATIBILITY

__all__ = ['activation_size', 'parameter_size', 'is_inplace']


def activation_size(out: Union[torch.Tensor, Dict, List, Tuple, int]) -> int:
    """Calculate activation size of a node.

    Args:
        activation (Union[torch.Tensor, Dict, List, Tuple, int]): The activation of a `torch.nn.Module` or `torch.nn.functional`

    Returns:
        int: The activation size
    """
    act_size = 0
    if isinstance(out, torch.Tensor):
        act_size += out.numel() * torch.tensor([], dtype=out.dtype).element_size()
    elif isinstance(out, dict):
        value_list = [v for _, v in out.items()]
        act_size += activation_size(value_list)
    elif isinstance(out, tuple) or isinstance(out, list):
        for element in out:
            act_size += activation_size(element)
    return act_size


def parameter_size(mod: torch.nn.Module) -> int:
    """Calculate parameter size of a node.

    Args:
        mod (torch.nn.Module): The target `torch.nn.Module`

    Returns:
        int: The parameter size
    """
    param_size = 0
    for param in mod.parameters():
        param_size += param.numel() * torch.tensor([], dtype=param.dtype).element_size()
    return param_size


def is_inplace(n: Node):
    """Get the inplace argument from torch.fx.Node

    Args:
        node (Node): torch.fx.Node

    Returns:
        bool: indicates whether this op is inplace
    """
    inplace = False
    if n.op == "call_function":
        inplace = n.kwargs.get("inplace", False)
        if META_COMPATIBILITY:
            from .constant import ALIAS_ATEN
            if n.target in ALIAS_ATEN:
                inplace = True
    elif n.op == "call_module":
        inplace = getattr(n.graph.owning_module.get_submodule(n.target), "inplace", False)

    return inplace
