import math
import warnings

from torch import Tensor
import torch.nn as nn


def zeros_():
    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        return nn.init.zeros_(tensor)

    return initializer


def ones_():
    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        return nn.init.ones_(tensor)

    return initializer


def uniform_(a: float = 0., b: float = 1.):
    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        return nn.init.uniform_(tensor, a, b)

    return initializer


def normal_(mean: float = 0., std: float = 1.):
    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        return nn.init.normal_(tensor, mean, std)

    return initializer


def trunc_normal_(mean: float = 0., std: float = 1., a: float = -2., b: float = 2.):
    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        return nn.init.trunc_normal_(tensor, mean, std, a, b)

    return initializer


def kaiming_uniform_(a=0, mode='fan_in', nonlinearity='leaky_relu'):
    # adapted from torch.nn.init
    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        if 0 in tensor.shape:
            warnings.warn("Initializing zero-element tensors is a no-op")
            return tensor

        if mode == 'fan_in':
            assert fan_in is not None, 'Fan_in is not provided.'
            fan = fan_in
        elif mode == 'fan_out':
            assert fan_out is not None, 'Fan_out is not provided.'
            fan = fan_out
        else:
            raise ValueError(f'Invalid initialization mode \'{mode}\'')

        std = nn.init.calculate_gain(nonlinearity, a) / math.sqrt(fan)
        bound = math.sqrt(3.) * std
        return nn.init.uniform_(tensor, -bound, bound)

    return initializer


def kaiming_normal_(a=0, mode='fan_in', nonlinearity='leaky_relu'):
    # adapted from torch.nn.init
    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        if 0 in tensor.shape:
            warnings.warn("Initializing zero-element tensors is a no-op")
            return tensor

        if mode == 'fan_in':
            assert fan_in is not None, 'Fan_in is not provided.'
            fan = fan_in
        elif mode == 'fan_out':
            assert fan_out is not None, 'Fan_out is not provided.'
            fan = fan_out
        else:
            raise ValueError(f'Invalid initialization mode \'{mode}\'')

        std = nn.init.calculate_gain(nonlinearity, a) / math.sqrt(fan)
        return nn.init.normal_(tensor, 0, std)

    return initializer


def xavier_uniform_(a: float = math.sqrt(3.), scale: float = 2., gain: float = 1.):
    # adapted from torch.nn.init
    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        assert fan_in is not None, 'Fan_in is not provided.'

        fan = fan_in
        if fan_out is not None:
            fan += fan_out

        std = gain * math.sqrt(scale / float(fan))
        bound = a * std
        return nn.init.uniform_(tensor, -bound, bound)

    return initializer


def xavier_normal_(scale: float = 2., gain: float = 1.):
    # adapted from torch.nn.init
    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        assert fan_in is not None, 'Fan_in is not provided.'

        fan = fan_in
        if fan_out is not None:
            fan += fan_out

        std = gain * math.sqrt(scale / float(fan))

        return nn.init.normal_(tensor, 0., std)

    return initializer


def lecun_uniform_():
    # adapted from jax.nn.initializers
    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        assert fan_in is not None, 'Fan_in is not provided.'

        var = 1.0 / fan_in
        bound = math.sqrt(3 * var)
        return nn.init.uniform_(tensor, -bound, bound)

    return initializer


def lecun_normal_():
    # adapted from jax.nn.initializers
    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        assert fan_in is not None, 'Fan_in is not provided.'

        std = math.sqrt(1.0 / fan_in)
        return nn.init.trunc_normal_(tensor, std=std / .87962566103423978)

    return initializer
