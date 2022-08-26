import operator
from functools import reduce
import math
from typing import Tuple
import torch
from ..registry import meta_profiler_module


@meta_profiler_module.register(torch.nn.Conv1d)
def torch_nn_conv1d(self: torch.nn.Conv1d, input: torch.Tensor) -> Tuple[int, int]:
    # the output shape is calculated using the formula stated
    # at https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    c_in, l_in = input.shape[-2:]
    c_out = self.out_channels
    l_out = math.floor((l_in + 2 * self.padding[0] - self.dilation[0] *
                        (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
    result_shape = input.shape[:-2] + (
        c_out,
        l_out,
    )
    macs_per_elem = reduce(operator.mul, self.kernel_size) * c_in // self.groups
    num_elem = reduce(operator.mul, result_shape)
    macs = macs_per_elem * num_elem
    flops = 2 * macs
    if self.bias is not None:
        flops += num_elem
    return flops, macs


@meta_profiler_module.register(torch.nn.Conv2d)
def torch_nn_conv2d(self: torch.nn.Conv2d, input: torch.Tensor) -> Tuple[int, int]:
    # the output shape is calculated using the formula stated
    # at https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    c_in, h_in, w_in = input.shape[-3:]
    c_out = self.out_channels
    h_out = math.floor((h_in + 2 * self.padding[0] - self.dilation[0] *
                        (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
    w_out = math.floor((w_in + 2 * self.padding[1] - self.dilation[1] *
                        (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
    result_shape = input.shape[:-3] + (
        c_out,
        h_out,
        w_out,
    )
    macs_per_elem = reduce(operator.mul, self.kernel_size) * c_in // self.groups
    num_elem = reduce(operator.mul, result_shape)
    macs = macs_per_elem * num_elem
    flops = 2 * macs
    if self.bias is not None:
        flops += num_elem
    return flops, macs


@meta_profiler_module.register(torch.nn.Conv3d)
def torch_nn_conv3d(self: torch.nn.Conv3d, input: torch.Tensor) -> Tuple[int, int]:
    # the output shape is calculated using the formula stated
    # at https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
    c_in, d_in, h_in, w_in = input.shape[-4:]
    c_out = self.out_channels
    d_out = math.floor((d_in + 2 * self.padding[0] - self.dilation[0] *
                        (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
    h_out = math.floor((h_in + 2 * self.padding[1] - self.dilation[1] *
                        (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
    w_out = math.floor((w_in + 2 * self.padding[2] - self.dilation[2] *
                        (self.kernel_size[2] - 1) - 1) / self.stride[2] + 1)
    result_shape = input.shape[:-4] + (
        c_out,
        d_out,
        h_out,
        w_out,
    )
    macs_per_elem = reduce(operator.mul, self.kernel_size) * c_in // self.groups
    num_elem = reduce(operator.mul, result_shape)
    macs = macs_per_elem * num_elem
    flops = 2 * macs
    if self.bias is not None:
        flops += num_elem
    return flops, macs


@meta_profiler_module.register(torch.nn.ConvTranspose1d)
def torch_nn_convtranspose1d(self: torch.nn.ConvTranspose1d, input: torch.Tensor) -> Tuple[int, int]:
    # the output shape is calculated using the formula stated
    # at https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
    c_in, l_in = input.shape[-2:]
    c_out = self.out_channels
    l_out = math.floor((l_in - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] *
                       (self.kernel_size[0] - 1) + self.output_padding[0] + 1)
    result_shape = input.shape[:-2] + (
        c_out,
        l_out,
    )
    macs_per_elem = reduce(operator.mul, self.kernel_size) * c_in // self.groups
    num_elem = reduce(
        operator.mul, input.shape
    )    # see https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/profiling/flops_profiler/profiler.py#L604
    macs = macs_per_elem * num_elem
    flops = 2 * macs
    if self.bias is not None:
        flops += reduce(operator.mul, result_shape)
    return flops, macs


@meta_profiler_module.register(torch.nn.ConvTranspose2d)
def torch_nn_convtranspose2d(self: torch.nn.ConvTranspose2d, input: torch.Tensor) -> Tuple[int, int]:
    # the output shape is calculated using the formula stated
    # at https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    c_in, h_in, w_in = input.shape[-3:]
    c_out = self.out_channels
    h_out = math.floor((h_in - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] *
                       (self.kernel_size[0] - 1) + self.output_padding[0] + 1)
    w_out = math.floor((w_in - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] *
                       (self.kernel_size[1] - 1) + self.output_padding[1] + 1)
    result_shape = input.shape[:-3] + (
        c_out,
        h_out,
        w_out,
    )
    macs_per_elem = reduce(operator.mul, self.kernel_size) * c_in // self.groups
    num_elem = reduce(operator.mul, input.shape)
    macs = macs_per_elem * num_elem
    flops = 2 * macs
    if self.bias is not None:
        flops += reduce(operator.mul, result_shape)
    return flops, macs


@meta_profiler_module.register(torch.nn.ConvTranspose3d)
def torch_nn_convtranspose3d(self: torch.nn.ConvTranspose3d, input: torch.Tensor) -> Tuple[int, int]:
    # the output shape is calculated using the formula stated
    # at https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html
    c_in, d_in, h_in, w_in = input.shape[-4:]
    c_out = self.out_channels
    d_out = math.floor((d_in - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] *
                       (self.kernel_size[0] - 1) + self.output_padding[0] + 1)
    h_out = math.floor((h_in - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] *
                       (self.kernel_size[1] - 1) + self.output_padding[1] + 1)
    w_out = math.floor((w_in - 1) * self.stride[2] - 2 * self.padding[2] + self.dilation[2] *
                       (self.kernel_size[2] - 1) + self.output_padding[2] + 1)
    result_shape = input.shape[:-4] + (
        c_out,
        d_out,
        h_out,
        w_out,
    )
    macs_per_elem = reduce(operator.mul, self.kernel_size) * c_in // self.groups
    num_elem = reduce(operator.mul, input.shape)
    macs = macs_per_elem * num_elem
    flops = 2 * macs
    if self.bias is not None:
        flops += reduce(operator.mul, result_shape)
    return flops, macs
