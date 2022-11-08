import math

import torch

from ...registry import meta_patched_module


@meta_patched_module.register(torch.nn.AvgPool1d)
def torch_nn_avgpool1d(self, input):
    num_dim = input.dim()
    assert num_dim in [2, 3], f'expected the input to have 2 or 3 dimensions, but got {num_dim} dimensions'

    l_in = input.shape[-1]

    def _convert_int_to_list(item):
        if isinstance(item, int):
            return [item] * 1
        else:
            return item

    padding = _convert_int_to_list(self.padding)
    kernel_size = _convert_int_to_list(self.kernel_size)
    stride = _convert_int_to_list(self.stride)

    l_out = math.floor((l_in + 2 * padding[0] - kernel_size[0]) / stride[0] + 1)

    result_shape = tuple(input.shape[:-1]) + (l_out,)
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.AvgPool2d)
def torch_nn_avgpool2d(self, input):
    num_dim = input.dim()
    assert num_dim in [3, 4], f'expected the input to have 3 or 4 dimensions, but got {num_dim} dimensions'

    h_in, w_in = input.shape[-2:]

    def _convert_int_to_list(item):
        if isinstance(item, int):
            return [item] * 2
        else:
            return item

    padding = _convert_int_to_list(self.padding)
    kernel_size = _convert_int_to_list(self.kernel_size)
    stride = _convert_int_to_list(self.stride)

    h_out = math.floor((h_in + 2 * padding[0] - kernel_size[0]) / stride[0] + 1)
    w_out = math.floor((w_in + 2 * padding[1] - kernel_size[1]) / stride[1] + 1)

    result_shape = tuple(input.shape[:-2]) + (
        h_out,
        w_out,
    )
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.AvgPool3d)
def torch_nn_avgpool3d(self, input):
    num_dim = input.dim()
    assert num_dim in [4, 5], f'expected the input to have 4 or 5 dimensions, but got {num_dim} dimensions'

    d_in, h_in, w_in = input.shape[-3:]

    def _convert_int_to_list(item):
        if isinstance(item, int):
            return [item] * 3
        else:
            return item

    padding = _convert_int_to_list(self.padding)
    kernel_size = _convert_int_to_list(self.kernel_size)
    stride = _convert_int_to_list(self.stride)

    d_out = math.floor((d_in + 2 * padding[0] - kernel_size[0]) / stride[0] + 1)
    h_out = math.floor((h_in + 2 * padding[1] - kernel_size[1]) / stride[1] + 1)
    w_out = math.floor((w_in + 2 * padding[2] - kernel_size[2]) / stride[2] + 1)

    result_shape = tuple(input.shape[:-3]) + (
        d_out,
        h_out,
        w_out,
    )
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.MaxPool1d)
def torch_nn_maxpool1d(self, input):
    num_dim = input.dim()
    assert num_dim in [2, 3], f'expected the input to have 2 or 3 dimensions, but got {num_dim} dimensions'

    l_in = input.shape[-1]

    def _convert_int_to_list(item):
        if isinstance(item, int):
            return [item] * 1
        else:
            return item

    padding = _convert_int_to_list(self.padding)
    dilation = _convert_int_to_list(self.dilation)
    kernel_size = _convert_int_to_list(self.kernel_size)
    stride = _convert_int_to_list(self.stride)

    l_out = math.floor((l_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)

    result_shape = tuple(input.shape[:-1]) + (l_out,)
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.MaxPool2d)
def torch_nn_maxpool2d(self, input):
    num_dim = input.dim()
    assert num_dim in [3, 4], f'expected the input to have 3 or 4 dimensions, but got {num_dim} dimensions'

    h_in, w_in = input.shape[-2:]

    def _convert_int_to_list(item):
        if isinstance(item, int):
            return [item] * 2
        else:
            return item

    padding = _convert_int_to_list(self.padding)
    dilation = _convert_int_to_list(self.dilation)
    kernel_size = _convert_int_to_list(self.kernel_size)
    stride = _convert_int_to_list(self.stride)

    h_out = math.floor((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w_out = math.floor((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    result_shape = tuple(input.shape[:-2]) + (
        h_out,
        w_out,
    )
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.MaxPool3d)
def torch_nn_maxpool3d(self, input):
    num_dim = input.dim()
    assert num_dim in [4, 5], f'expected the input to have 4 or 5 dimensions, but got {num_dim} dimensions'

    d_in, h_in, w_in = input.shape[-3:]

    def _convert_int_to_list(item):
        if isinstance(item, int):
            return [item] * 3
        else:
            return item

    padding = _convert_int_to_list(self.padding)
    dilation = _convert_int_to_list(self.dilation)
    kernel_size = _convert_int_to_list(self.kernel_size)
    stride = _convert_int_to_list(self.stride)

    d_out = math.floor((d_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    h_out = math.floor((h_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    w_out = math.floor((w_in + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2] + 1)

    result_shape = tuple(input.shape[:-3]) + (
        d_out,
        h_out,
        w_out,
    )
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.AdaptiveAvgPool1d)
@meta_patched_module.register(torch.nn.AdaptiveMaxPool1d)
def torch_nn_adapative_pooling_1d(self, input):
    assert input.dim() in [2, 3]
    if isinstance(self.output_size, int):
        output_size = (self.output_size,)
    else:
        output_size = self.output_size
    result_shape = tuple(input.shape[:-1]) + output_size
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.AdaptiveAvgPool2d)
@meta_patched_module.register(torch.nn.AdaptiveMaxPool2d)
def torch_nn_adapative_pooling_2d(self, input):
    assert input.dim() in [3, 4]
    if isinstance(self.output_size, int):
        output_size = (self.output_size,) * 2
    else:
        output_size = self.output_size
    result_shape = tuple(input.shape[:-2]) + output_size
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.AdaptiveAvgPool3d)
@meta_patched_module.register(torch.nn.AdaptiveMaxPool3d)
def torch_nn_adapative_pooling_3d(self, input):
    assert input.dim() in [4, 5]
    if isinstance(self.output_size, int):
        output_size = (self.output_size,) * 3
    else:
        output_size = self.output_size
    result_shape = tuple(input.shape[:-3]) + output_size
    return torch.empty(result_shape, device='meta')
