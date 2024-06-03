from operator import add, floordiv, getitem, mul, neg, pos, setitem, sub

import torch

__all__ = ["INPLACE_OPS", "INPLACE_METHOD", "NON_INPLACE_METHOD"]

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
    "transpose",
    "permute",
    # TODO: reshape may return a copy of the data if the data is not contiguous
    "reshape",
    "dim",
    "flatten",
    "size",
    "view",
    "unsqueeze",
    "to",
    "type",
    "flatten",
]

# TODO: list all call_methods that are not inplace here
NON_INPLACE_METHOD = [
    "chunk",
    "contiguous",
    "expand",
    "mean",
    "split",
]
