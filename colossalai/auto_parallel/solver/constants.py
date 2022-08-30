import torch
import operator

__all__ = [
    'ELEMENTWISE_MODULE_OP', 'ELEMENTWISE_FUNC_OP', 'CONV_MODULE_OP', 'CONV_FUNC_OP', 'LINEAR_MODULE_OP',
    'LINEAR_FUNC_OP'
]

ELEMENTWISE_MODULE_OP = [torch.nn.Dropout, torch.nn.ReLU]
ELEMENTWISE_FUNC_OP = [
    torch.add, operator.add, torch.abs, torch.cos, torch.exp, torch.mul, operator.mul, operator.floordiv,
    operator.truediv, operator.neg, torch.multiply, torch.nn.functional.relu, torch.nn.functional.dropout
]
CONV_MODULE_OP = [
    torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d
]
CONV_FUNC_OP = [
    torch.conv1d, torch.conv2d, torch.conv3d, torch.conv_transpose1d, torch.conv_transpose2d, torch.conv_transpose3d
]
LINEAR_MODULE_OP = [torch.nn.Linear]
LINEAR_FUNC_OP = [torch.nn.functional.linear, torch.matmul, torch.bmm]
