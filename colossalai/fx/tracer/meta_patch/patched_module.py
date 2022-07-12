import math
import torch
from .registry import meta_patched_module


@meta_patched_module.register(torch.nn.Linear)
def torch_nn_linear(self, input):
    last_dim = input.shape[-1]
    assert last_dim == self.in_features, f'Expected hidden size {self.in_features} but got {last_dim} for the torch.nn.Linear patch'
    return torch.empty(input.shape[:-1] + (self.out_features,), device="meta")


@meta_patched_module.register(torch.nn.LayerNorm)
@meta_patched_module.register(torch.nn.GroupNorm)
@meta_patched_module.register(torch.nn.BatchNorm1d)
@meta_patched_module.register(torch.nn.BatchNorm2d)
@meta_patched_module.register(torch.nn.BatchNorm3d)
def torch_nn_normalize(self, input):
    # check shape
    if isinstance(self, torch.nn.BatchNorm1d):
        assert input.dim() in [2, 3]
    elif isinstance(self, torch.nn.BatchNorm2d):
        assert input.dim() == 4
    elif isinstance(self, torch.nn.BatchNorm3d):
        assert input.dim() == 5

    # normalization maintain the same shape as the input
    return input.clone()


@meta_patched_module.register(torch.nn.Embedding)
def torch_nn_embedding(self, input):
    result_shape = input.shape + (self.embedding_dim,)
    return torch.empty(result_shape, device='meta')


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

    result_shape = input.shape[:-1] + (l_out,)
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

    result_shape = input.shape[:-2] + (
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

    result_shape = input.shape[:-3] + (
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

    result_shape = input.shape[:-1] + (l_out,)
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

    result_shape = input.shape[:-2] + (
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

    result_shape = input.shape[:-3] + (
        d_out,
        h_out,
        w_out,
    )
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.AdaptiveAvgPool1d)
@meta_patched_module.register(torch.nn.AdaptiveMaxPool1d)
def torch_nn_adapative_pooling_1d(self, input):
    result_shape = input.shape[:-1] + (self.output_size,)
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.AdaptiveAvgPool2d)
@meta_patched_module.register(torch.nn.AdaptiveMaxPool2d)
def torch_nn_adapative_pooling_2d(self, input):
    result_shape = input.shape[:-2] + (
        self.output_size,
        self.output_size,
    )
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.AdaptiveAvgPool3d)
@meta_patched_module.register(torch.nn.AdaptiveMaxPool3d)
def torch_nn_adapative_pooling_3d(self, input):
    result_shape = input.shape[:-3] + (
        self.output_size,
        self.output_size,
        self.output_size,
    )
    return torch.empty(result_shape, device='meta')


@meta_patched_module.register(torch.nn.ReLU)
@meta_patched_module.register(torch.nn.ReLU6)
def torch_nn_func_relu(self, input):
    return torch.empty(input.shape, device='meta')
