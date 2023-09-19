"""
If FX.Graph is traced for auto-parallel module, some extra node will be added during
graph construction to deal with the compatibility between bias-addition and all-reduce.
"""

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _single, _triple

from .tracer import register_tracer_impl

__all__ = []


@register_tracer_impl(F.linear, name="_bias_addition_impl")
def linear_impl(input, weight, bias=None):
    if bias is None:
        return F.linear(input, weight)
    else:
        return F.linear(input, weight) + bias


@register_tracer_impl(F.conv1d, name="_bias_addition_impl")
def conv1d_impl(input, weight, bias=None, stride=_single(1), padding=_single(0), dilation=_single(1), groups=1):
    if bias is None:
        return F.conv1d(input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
    else:
        return F.conv1d(input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups) + bias.reshape(
            (-1, 1)
        )


@register_tracer_impl(F.conv2d, name="_bias_addition_impl")
def conv2d_impl(input, weight, bias=None, stride=_pair(1), padding=_pair(0), dilation=_pair(1), groups=1):
    if bias is None:
        return F.conv2d(input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
    else:
        return F.conv2d(input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups) + bias.reshape(
            (-1, 1, 1)
        )


@register_tracer_impl(F.conv3d, name="_bias_addition_impl")
def conv3d_impl(input, weight, bias=None, stride=_triple(1), padding=_triple(0), dilation=_triple(1), groups=1):
    if bias is None:
        return F.conv3d(input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
    else:
        return F.conv3d(input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups) + bias.reshape(
            (-1, 1, 1, 1)
        )


@register_tracer_impl(F.conv_transpose1d, name="_bias_addition_impl")
def conv_transpose1d_impl(
    input,
    weight,
    bias=None,
    stride=_single(1),
    padding=_single(0),
    output_padding=_single(0),
    groups=1,
    dilation=_single(1),
):
    if bias is None:
        return F.conv_transpose1d(
            input,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )
    else:
        return F.conv_transpose1d(
            input,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        ) + bias.reshape((-1, 1))


@register_tracer_impl(F.conv_transpose2d, name="_bias_addition_impl")
def conv_transpose2d_impl(
    input, weight, bias=None, stride=_pair(1), padding=_pair(0), output_padding=_pair(0), groups=1, dilation=_pair(1)
):
    if bias is None:
        return F.conv_transpose2d(
            input,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )
    else:
        return F.conv_transpose2d(
            input,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        ) + bias.reshape((-1, 1, 1))


@register_tracer_impl(F.conv_transpose3d, name="_bias_addition_impl")
def conv_transpose3d_impl(
    input,
    weight,
    bias=None,
    stride=_triple(1),
    padding=_triple(0),
    output_padding=_triple(0),
    groups=1,
    dilation=_triple(1),
):
    if bias is None:
        return F.conv_transpose3d(
            input,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )
    else:
        return F.conv_transpose3d(
            input,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        ) + bias.reshape((-1, 1, 1, 1))


@register_tracer_impl(torch.addmm, name="_bias_addition_impl")
@register_tracer_impl(torch.Tensor.addmm, name="_bias_addition_impl")
def addmm_impl(input, mat1, mat2, beta=1, alpha=1):
    if alpha != 1 and beta != 1:
        return F.linear(mat1, mat2.transpose(0, 1)) * alpha + input * beta
    elif alpha != 1:
        return F.linear(mat1, mat2.transpose(0, 1)) * alpha + input
    elif beta != 1:
        return F.linear(mat1, mat2.transpose(0, 1)) + input * beta
    else:
        return F.linear(mat1, mat2.transpose(0, 1)) + input


@register_tracer_impl(torch.addbmm, name="_bias_addition_impl")
@register_tracer_impl(torch.Tensor.addbmm, name="_bias_addition_impl")
def addbmm_impl(input, batch1, batch2, beta=1, alpha=1):
    if alpha != 1 and beta != 1:
        return torch.bmm(batch1, batch2.transpose(1, 2)) * alpha + input * beta
    elif alpha != 1:
        return torch.bmm(batch1, batch2.transpose(1, 2)) * alpha + input
    elif beta != 1:
        return torch.bmm(batch1, batch2.transpose(1, 2)) + input * beta
    else:
        return torch.bmm(batch1, batch2.transpose(1, 2)) + input
