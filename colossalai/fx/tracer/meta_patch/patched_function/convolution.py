import collections
import math
from itertools import repeat

import torch

from ...registry import meta_patched_function


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")


def _extract_kwargs(kwargs):
    if "stride" in kwargs:
        stride = kwargs["stride"]
    else:
        stride = 1
    # TODO: process str type padding
    if "padding" in kwargs:
        padding = kwargs["padding"]
    else:
        padding = 0
    if "dilation" in kwargs:
        dilation = kwargs["dilation"]
    else:
        dilation = 1
    if "output_padding" in kwargs:
        output_padding = kwargs["output_padding"]
    else:
        output_padding = 0

    return stride, padding, dilation, output_padding


@meta_patched_function.register(torch.nn.functional.conv1d)
def torch_nn_functional_conv1d(input, weight, **kwargs):
    stride, padding, dilation, _ = _extract_kwargs(kwargs)

    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)

    kernel_size = weight.shape[2:]
    l_in = input.shape[-1]
    c_out = weight.shape[0]
    l_out = math.floor((l_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    result_shape = input.shape[:-2] + (
        c_out,
        l_out,
    )
    return torch.empty(result_shape, device="meta")


@meta_patched_function.register(torch.nn.functional.conv2d)
def torch_nn_functional_conv2d(input, weight, **kwargs):
    stride, padding, dilation, _ = _extract_kwargs(kwargs)

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    kernel_size = weight.shape[2:]
    h_in, w_in = input.shape[-2:]
    c_out = weight.shape[0]
    h_out = math.floor((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w_out = math.floor((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    result_shape = input.shape[:-3] + (
        c_out,
        h_out,
        w_out,
    )
    return torch.empty(result_shape, device="meta")


@meta_patched_function.register(torch.nn.functional.conv3d)
def torch_nn_functional_conv3d(input, weight, **kwargs):
    stride, padding, dilation, _ = _extract_kwargs(kwargs)

    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    kernel_size = weight.shape[2:]
    d_in, h_in, w_in = input.shape[-3:]
    c_out = weight.shape[0]
    d_out = math.floor((d_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    h_out = math.floor((h_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    w_out = math.floor((w_in + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2] + 1)
    result_shape = input.shape[:-4] + (
        c_out,
        d_out,
        h_out,
        w_out,
    )
    return torch.empty(result_shape, device="meta")


@meta_patched_function.register(torch.nn.functional.conv_transpose1d)
def torch_nn_functional_convtranspose1d(input, weight, **kwargs):
    stride, padding, dilation, output_padding = _extract_kwargs(kwargs)

    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)
    output_padding = _single(output_padding)

    kernel_size = weight.shape[2:]
    l_in = input.shape[-1]
    c_out = weight.shape[1]
    l_out = math.floor(
        (l_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
    )
    result_shape = input.shape[:-2] + (
        c_out,
        l_out,
    )
    return torch.empty(result_shape, device="meta")


@meta_patched_function.register(torch.nn.functional.conv_transpose2d)
def torch_nn_functional_convtranspose2d(input, weight, **kwargs):
    stride, padding, dilation, output_padding = _extract_kwargs(kwargs)

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    output_padding = _pair(output_padding)

    kernel_size = weight.shape[2:]
    h_in, w_in = input.shape[-2:]
    c_out = weight.shape[1]
    h_out = math.floor(
        (h_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
    )
    w_out = math.floor(
        (w_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1
    )
    result_shape = input.shape[:-3] + (
        c_out,
        h_out,
        w_out,
    )
    return torch.empty(result_shape, device="meta")


@meta_patched_function.register(torch.nn.functional.conv_transpose3d)
def torch_nn_functional_convtranspose3d(input, weight, **kwargs):
    stride, padding, dilation, output_padding = _extract_kwargs(kwargs)

    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)
    output_padding = _triple(output_padding)

    kernel_size = weight.shape[2:]
    d_in, h_in, w_in = input.shape[-3:]
    c_out = weight.shape[1]
    d_out = math.floor(
        (d_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
    )
    h_out = math.floor(
        (h_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1
    )
    w_out = math.floor(
        (w_in - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_size[2] - 1) + output_padding[2] + 1
    )
    result_shape = input.shape[:-4] + (
        c_out,
        d_out,
        h_out,
        w_out,
    )
    return torch.empty(result_shape, device="meta")
