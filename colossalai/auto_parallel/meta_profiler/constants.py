import operator

import torch
import torch.nn as nn

# list of inplace operations
INPLACE_MODULE = [nn.ReLU]

# list of operations that do not save forward activations
NO_SAVE_ACTIVATION = [torch.add, torch.sub, operator.add, operator.sub]

# list of binary elementwise operations
BCAST_FUNC_OP = [
    torch.add, torch.sub, torch.mul, torch.div, torch.floor_divide, torch.true_divide, operator.add, operator.sub,
    operator.mul, operator.floordiv, operator.truediv, torch.matmul, operator.pow, torch.pow
]
