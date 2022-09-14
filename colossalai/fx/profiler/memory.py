import torch
from typing import Union, Dict, List, Tuple
from operator import add, floordiv, getitem, mul, neg, setitem, sub, pos
from . import META_COMPATIBILITY

__all__ = ['activation_size', 'parameter_size']

if META_COMPATIBILITY:
    aten = torch.ops.aten

    WEIRD_OPS = [
        torch.where,
    ]

    INPLACE_ATEN = [
        aten.add_.Tensor,
        aten.sub_.Tensor,
        aten.div_.Tensor,
        aten.div_.Scalar,
        aten.mul_.Tensor,
        aten.bernoulli_.float,

    # inplace reshaping
        aten.detach.default,
        aten.t.default,
        aten.transpose.int,
        aten.view.default,
        aten._unsafe_view.default,
    ]

    __all__ += ['INPLACE_ATEN', 'WEIRD_OPS']

else:
    # TODO fill out the inplace ops
    INPLACE_OPS = [
        add,
        sub,
        mul,
        floordiv,
        neg,
        pos,
        getitem,
        setitem,
        getattr,
        torch.Tensor.cpu,
    ]

    # TODO: list all call_methods that are inplace here
    INPLACE_METHOD = [
        'transpose',
        'permute',
    # TODO: reshape may return a copy of the data if the data is not contiguous
        'reshape',
        'dim',
        'flatten',
        'size',
        'view',
        'unsqueeze',
        'to',
        'type',
        'flatten',
    ]

    # TODO: list all call_methods that are not inplace here
    NON_INPLACE_METHOD = [
        'chunk',
        'contiguous',
        'expand',
        'mean',
        'split',
    ]
    __all__ += ['INPLACE_OPS', 'INPLACE_METHOD', 'NON_INPLACE_METHOD']


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
    """Calculate param size of a node.

    Args:
        mod (torch.nn.Module): The target `torch.nn.Module`

    Returns:
        int: The param size
    """
    param_size = 0
    for param in mod.parameters():
        param_size += param.numel() * torch.tensor([], dtype=param.dtype).element_size()
    return param_size
