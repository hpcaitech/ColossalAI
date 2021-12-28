from typing import Optional

import torch.nn as nn
from colossalai.context import ParallelMode, seed
from colossalai.utils import conditional_context

from ... import init as init
from ..parallel_1d import *
from ..parallel_2d import *
from ..parallel_2p5d import *
from ..parallel_3d import *
from ..utils import get_tensor_parallel_mode
from ..vanilla import *


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        self.tensor_parallel = get_tensor_parallel_mode()
        if self.tensor_parallel == '1d':
            self.drop = Dropout1D(p, inplace)
        else:
            self.drop = nn.Dropout(p, inplace)

    def forward(self, *args):
        with conditional_context(seed(ParallelMode.TENSOR), enable=self.tensor_parallel not in ['None', '1d']):
            return self.drop(*args)
