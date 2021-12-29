from contextlib import nullcontext

import torch.nn as nn
from colossalai.context import ParallelMode, seed
from colossalai.utils import conditional_context

from ..parallel_1d import *
from ..utils import get_tensor_parallel_mode


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        self.tensor_parallel = get_tensor_parallel_mode()
        if self.tensor_parallel == '1d':
            self.drop = Dropout1D(p, inplace)
        else:
            self.drop = nn.Dropout(p, inplace)

    def forward(self, *args):
        cm = nullcontext() if self.tensor_parallel in ['None', '1d'] else seed(ParallelMode.TENSOR)
        with cm:
            return self.drop(*args)
