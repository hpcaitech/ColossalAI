import torch
from operator import add, floordiv, getitem, mul, neg, setitem, sub, pos
from . import META_COMPATIBILITY

__all__ = []

if META_COMPATIBILITY:
    aten = torch.ops.aten

    ALIAS_ATEN = [
    # inplace reshaping
        aten.detach.default,
        aten.t.default,
        aten.transpose.int,
        aten.view.default,
        aten._unsafe_view.default,
    ]

    INPLACE_NEW = [
        aten.empty_like.default,
        aten.new_empty_strided.default,
    ]

    INPLACE_MATH_ATEN = [
        aten.add_.Tensor,
        aten.sub_.Tensor,
        aten.div_.Tensor,
        aten.div_.Scalar,
        aten.mul_.Tensor,
        aten.bernoulli_.float,
    ]

    CLONE_ATEN = [
        aten.clone.default,
    ]

    __all__ += ['INPLACE_ATEN', 'INPLACE_MATH_ATEN', 'CLONE_ATEN']

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
