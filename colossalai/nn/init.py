import math
import warnings

from torch import Tensor
import torch.nn as nn


def zeros_():
    """Return the initializer filling the input Tensor with the scalar zeros"""
    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        return nn.init.zeros_(tensor)

    return initializer


def ones_():
    """Return the initializer filling the input Tensor with the scalar ones"""
    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        return nn.init.ones_(tensor)

    return initializer


def uniform_(a: float = 0., b: float = 1.):
    r"""Return the initializer filling the input Tensor with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.

    :param a: the lower bound of the uniform distribution
    :param b: the upper bound of the uniform distribution
    :type a: float
    :type b: float
    """

    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        return nn.init.uniform_(tensor, a, b)

    return initializer


def normal_(mean: float = 0., std: float = 1.):
    r"""Return the initializer filling the input Tensor with values drawn from the normal distribution

     .. math::
        \mathcal{N}(\text{mean}, \text{std}^2)

    :param mean: the mean of the normal distribution
    :param std: the standard deviation of the normal distribution
    :type mean: float
    :type std: float
     """
    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        return nn.init.normal_(tensor, mean, std)

    return initializer


def trunc_normal_(mean: float = 0., std: float = 1., a: float = -2., b: float = 2.):
    r"""Return the initializer filling the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    :param mean: the mean of the normal distribution
    :param std: the standard deviation of the normal distribution
    :param a: the minimum cutoff value
    :param b: the maximum cutoff value
    :type mean: float
    :type std: float
    :type a: float
    :type b: float
    """
    def initializer(tensor: Tensor, fan_in: int = None, fan_out: int = None):
        return nn.init.trunc_normal_(tensor, mean, std, a, b)

    return initializer


def kaiming_uniform_(a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Return the initializer filling the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    :param a: the negative slope of the rectifier used after this layer (only used with ``'leaky_relu'``)
    :param mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
                preserves the magnitude of the variance of the weights in the
                forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
                backwards pass.
    :param nonlinearity: the non-linear function (`nn.functional` name),
                        recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
    :type a: int
    :type mode: str
    :type nonlinearity: str
    """
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
    r"""Return the initializer filling the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Also known as He initialization.

    :param a: the negative slope of the rectifier used after this layer (only used with ``'leaky_relu'``)
    :param mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
                preserves the magnitude of the variance of the weights in the
                forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
                backwards pass.
    :param nonlinearity: the non-linear function (`nn.functional` name),
                        recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
    :type a: int
    :type mode: str
    :type nonlinearity: str
    """
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
    r"""Return the initializer filling the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    :param a: an optional scaling factor used to calculate uniform bounds from standard deviation
    :param scale: an optional scaling factor used to calculate standard deviation
    :param gain: an optional scaling factor
    :type a: float
    :type scale: float
    :type gain: float
    """
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
    r"""Return the initializer filling the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.
    :param scale: an optional scaling factor used to calculate standard deviation
    :param gain: an optional scaling factor
    :type scale: float
    :type gain: float
    """
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
