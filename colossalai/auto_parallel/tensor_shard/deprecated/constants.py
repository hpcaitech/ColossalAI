import torch
import operator

__all__ = [
    'ELEMENTWISE_MODULE_OP', 'ELEMENTWISE_FUNC_OP', 'RESHAPE_FUNC_OP', 'CONV_MODULE_OP', 'CONV_FUNC_OP',
    'LINEAR_MODULE_OP', 'LINEAR_FUNC_OP', 'BATCHNORM_MODULE_OP', 'POOL_MODULE_OP', 'NON_PARAM_FUNC_OP', 'BCAST_FUNC_OP',
    'EMBEDDING_MODULE_OP', 'LAYERNORM_MODULE_OP', 'ELEMENTWISE_METHOD_OP', 'RESHAPE_METHOD_OP', 'INFINITY_COST'
]

ELEMENTWISE_MODULE_OP = [torch.nn.Dropout, torch.nn.ReLU]
ELEMENTWISE_FUNC_OP = [
    torch.abs,
    torch.cos,
    torch.exp,
    operator.neg,
    torch.multiply,
    torch.nn.functional.relu,
    torch.nn.functional.dropout,
    # softmax should not be here
    torch.nn.functional.softmax
]
ELEMENTWISE_METHOD_OP = [
    torch.Tensor.to,
    torch.Tensor.type,
    # TODO: contiguous maybe need some extra processes.
    torch.Tensor.contiguous
]
RESHAPE_FUNC_OP = [torch.flatten, torch.reshape]
RESHAPE_METHOD_OP = [
    torch.Tensor.view,
    torch.Tensor.unsqueeze,
    torch.Tensor.split,
    torch.Tensor.permute,
    torch.Tensor.transpose,
]
BCAST_FUNC_OP = [
    torch.add, torch.sub, torch.mul, torch.div, torch.floor_divide, torch.true_divide, operator.add, operator.sub,
    operator.mul, operator.floordiv, operator.truediv, torch.matmul, torch.where, operator.pow, torch.pow, torch.tanh
]
CONV_MODULE_OP = [
    torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d
]
CONV_FUNC_OP = [
    torch.conv1d, torch.conv2d, torch.conv3d, torch.conv_transpose1d, torch.conv_transpose2d, torch.conv_transpose3d
]
EMBEDDING_MODULE_OP = [torch.nn.modules.sparse.Embedding]
LINEAR_MODULE_OP = [torch.nn.Linear]
LINEAR_FUNC_OP = [torch.nn.functional.linear, torch.matmul, torch.bmm]
BATCHNORM_MODULE_OP = [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm]
LAYERNORM_MODULE_OP = [torch.nn.LayerNorm]
POOL_MODULE_OP = [torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d, torch.nn.AdaptiveAvgPool2d]
NON_PARAM_FUNC_OP = [
    torch.flatten,
    torch.reshape,
    torch.abs,
    torch.cos,
    torch.exp,
    operator.neg,
    torch.multiply,
    torch.nn.functional.relu,
    torch.nn.functional.dropout,
    torch.flatten,
    torch.where,
    operator.pow,
    torch.pow,
    torch.tanh,
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
    torch.floor_divide,
    torch.true_divide,
    operator.add,
    operator.sub,
    operator.mul,
    operator.floordiv,
    operator.truediv,
    # softmax should not be here
    torch.nn.functional.softmax
]

INFINITY_COST = 1e13
