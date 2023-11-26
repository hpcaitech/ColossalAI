import operator

import torch
import torch.nn as nn

# list of inplace module
INPLACE_MODULE = [nn.ReLU]

# list of inplace operations
INPLACE_OPS = [torch.flatten]

# list of operations that do not save forward activations
NO_SAVE_ACTIVATION = [torch.add, torch.sub, operator.add, operator.sub]
