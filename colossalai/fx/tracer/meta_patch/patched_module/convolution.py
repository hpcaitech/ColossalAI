import math

import torch

from ...registry import meta_patched_module


@meta_patched_module.register(torch.nn.Conv1d)
def torch_nn_conv1d(self, input):
    # the output shape is calculated using the formula stated
    # at https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
    l_in = input.shape[-1]
    c_out = self.out_channels
    l_out = math.floor((l_in + 2 * self.padding[0] - self.dilation[0] *
                        (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
    result_shape = input.shape[:-2] + (
        c_out,
        l_out,
    )
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.Conv2d)
def torch_nn_conv2d(self, input):
    # the output shape is calculated using the formula stated
    # at https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv2d
    h_in, w_in = input.shape[-2:]
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
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.Conv3d)
def torch_nn_conv3d(self, input):
    # the output shape is calculated using the formula stated
    # at https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv3d
    d_in, h_in, w_in = input.shape[-3:]
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
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.ConvTranspose1d)
def torch_nn_convtranspose1d(self, input):
    # the output shape is calculated using the formula stated
    # at https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
    l_in = input.shape[-1]
    c_out = self.out_channels
    l_out = math.floor((l_in - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] *
                       (self.kernel_size[0] - 1) + self.output_padding[0] + 1)
    result_shape = input.shape[:-2] + (
        c_out,
        l_out,
    )
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.ConvTranspose2d)
def torch_nn_convtranspose2d(self, input):
    # the output shape is calculated using the formula stated
    # at https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    h_in, w_in = input.shape[-2:]
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
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.ConvTranspose3d)
def torch_nn_convtranspose3d(self, input):
    # the output shape is calculated using the formula stated
    # at https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html
    d_in, h_in, w_in = input.shape[-3:]
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
    return torch.empty(result_shape, device='meta')
