import torch
import torch.distributed as dist

aten = torch.ops.aten

__all__ = [
    'TorchFactoryMethod', 'TorchOverrideableFactoryMethod', 'TorchNonOverrideableFactoryMethod', 'TensorPropertyMethod',
    'DistCommMethod', 'AliasATen', 'InplaceATen', 'MaybeInplaceAten', 'SameStorageAten'
]

TorchOverrideableFactoryMethod = [
    'empty',
    'eye',
    'full',
    'ones',
    'rand',
    'randn',
    'zeros',
]

TorchNonOverrideableFactoryMethod = [
    'arange',
    'finfo',
    'linspace',
    'logspace',
    'randint',
    'randperm',
    'tensor',
]

TorchFactoryMethod = TorchOverrideableFactoryMethod + TorchNonOverrideableFactoryMethod

TensorPropertyMethod = ['dtype', 'shape', 'device', 'requires_grad', 'grad', 'grad_fn', 'data']

DistCommMethod = [
    'all_gather',
    'all_reduce',
    'all_to_all',
    'broadcast',
    'gather',
    'reduce',
    'reduce_scatter',
    'scatter',
]

AliasATen = [
    aten.detach.default,
    aten.detach_.default,
    aten.t.default,
    aten.transpose.int,
    aten.view.default,
    aten._unsafe_view.default,
    aten._reshape_alias.default,
]

InplaceATen = [
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

MaybeInplaceAten = [
    aten.diagonal.default,
    aten.select.int,
    aten.slice.Tensor,
    aten.as_strided.default,
]

SameStorageAten = AliasATen + InplaceATen + MaybeInplaceAten
