import torch
from torch.fx import Node, GraphModule
from typing import Union, Dict, List, Tuple
from . import META_COMPATIBILITY

__all__ = [
    'activation_size', 'parameter_size', 'is_inplace', "calculate_fwd_in", "calculate_fwd_tmp", "calculate_fwd_out"
]


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
    elif isinstance(out, tuple) or isinstance(out, list) or isinstance(out, set):
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


def calculate_fwd_in(n: Node) -> int:
    """A helper function to calculate `fwd_in`

    Args:
        n (Node): a node from the graph

    Returns:
        fwd_in (int): the result of `fwd_in`
    """
    return activation_size(n.meta["fwd_in"])


def calculate_fwd_tmp(n: Node) -> int:
    """A helper function to calculate `fwd_tmp`
    Currently, `torch.nn.ReLU` behaves weirdly, so we have to patch it for accuracy.

    Args:
        n (Node): a node from the graph

    Returns:
        fwd_tmp (int): the result of `fwd_tmp`
    """

    def is_relu_node(n: Node) -> bool:
        if n.op == 'call_function':
            return n.target in [torch.nn.functional.relu]
        elif n.op == 'call_module':
            return type(n.graph.owning_module.get_submodule(n.target)) in [torch.nn.ReLU]
        return False

    if not is_relu_node(n):
        return activation_size(n.meta["fwd_tmp"])
    return 0


def calculate_fwd_out(n: Node) -> int:
    """A helper function to calculate `fwd_out`

    Args:
        n (Node): a node from the graph

    Returns:
        fwd_out (int): the result of `fwd_out`
    """

    def intersect(a, b):
        return {k: a[k] for k in a if k in b}

    fwd_in = dict()
    for u in n.users:
        fwd_in.update({x.uuid: x for x in u.meta["fwd_in"] if isinstance(x, torch.Tensor) and hasattr(x, 'uuid')})
    fwd_out = {x.uuid: x for x in n.meta["fwd_out"] if isinstance(x, torch.Tensor) and hasattr(x, 'uuid')}
    return activation_size(intersect(fwd_in, fwd_out))


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
