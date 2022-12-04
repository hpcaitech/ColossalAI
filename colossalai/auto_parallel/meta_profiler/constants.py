import operator

import torch
import torch.nn as nn

from ..tensor_shard.constants import *

# list of inplace operations
INPLACE_MODULE = [nn.ReLU]

# list of operations that do not save forward activations
NO_SAVE_ACTIVATION = [torch.add, torch.sub, operator.add, operator.sub]
