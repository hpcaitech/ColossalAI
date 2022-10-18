import torch

__all__ = ['ALIAS_ATEN', 'INPLACE_NEW', 'INPLACE_MATH_ATEN', 'CLONE_ATEN']

aten = torch.ops.aten

ALIAS_ATEN = [
    aten.detach.default,
    aten.t.default,
    aten.transpose.int,
    aten.view.default,
    aten._unsafe_view.default,
    aten._reshape_alias.default,
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
