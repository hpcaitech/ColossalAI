import torch
from packaging import version

__all__ = [
    "_TorchFactoryMethod",
    "_TorchOverrideableFactoryMethod",
    "_TorchNonOverrideableFactoryMethod",
    "_TensorPropertyMethod",
    "_DistCommMethod",
    "_AliasATen",
    "_InplaceATen",
    "_MaybeInplaceATen",
]

_TorchOverrideableFactoryMethod = [
    "empty",
    "eye",
    "full",
    "ones",
    "rand",
    "randn",
    "zeros",
]

_TorchNonOverrideableFactoryMethod = [
    "arange",
    "finfo",
    "linspace",
    "logspace",
    "randint",
    "randperm",
    "tensor",
]

_TorchFactoryMethod = _TorchOverrideableFactoryMethod + _TorchNonOverrideableFactoryMethod

_TensorPropertyMethod = ["dtype", "shape", "device", "requires_grad", "grad", "grad_fn", "data"]

_DistCommMethod = [
    "all_gather",
    "all_reduce",
    "all_to_all",
    "broadcast",
    "gather",
    "reduce",
    "reduce_scatter",
    "scatter",
]

if version.parse(torch.__version__) >= version.parse("1.12.0"):
    aten = torch.ops.aten
    # TODO: dive deep here
    # refer to https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorShape.cpp
    _AliasATen = [
        aten.detach.default,
        aten.detach_.default,
        aten.t.default,
        aten.transpose.int,
        aten.view.default,
        aten._unsafe_view.default,
        aten._reshape_alias.default,
    ]

    _InplaceATen = [
        aten.add_.Tensor,
        aten.add_.Scalar,
        aten.sub_.Tensor,
        aten.sub_.Scalar,
        aten.mul_.Tensor,
        aten.mul_.Scalar,
        aten.div_.Tensor,
        aten.div_.Scalar,
        aten.pow_.Tensor,
        aten.pow_.Scalar,
    ]

    # use `MaybeInplace` because they call ``as_strided()`` or ``slice()``
    _MaybeInplaceATen = [
        aten.diagonal.default,
        aten.expand.default,
        aten.select.int,
        aten.slice.Tensor,
        aten.split.Tensor,
        aten.squeeze.default,
        aten.permute.default,
        aten.unsqueeze.default,
        aten.as_strided.default,
    ]
else:
    _AliasATen = []
    _InplaceATen = []
    _MaybeInplaceATen = []
