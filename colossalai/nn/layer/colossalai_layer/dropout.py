import torch.nn as nn
from colossalai.context import ParallelMode, seed

from ..parallel_1d import *
from ..utils import get_tensor_parallel_mode


class Dropout(nn.Module):
    """
    Dropout layer of colossalai

    :param p: dropout rate, defaults to 0.5
    :type p: float, optional
    :param inplace: If set to ``True``, will do this operation in-place, defaults tp ``False``
    :type inplace: bool, optional
    """
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        self.tensor_parallel = get_tensor_parallel_mode()
        if self.tensor_parallel == '1d':
            self.drop = Dropout1D(p, inplace)
        else:
            self.drop = nn.Dropout(p, inplace)

    def forward(self, *args):
        if self.tensor_parallel in [None, '1d']:
            return self.drop(*args)
        else:
            with seed(ParallelMode.TENSOR):
                return self.drop(*args)
