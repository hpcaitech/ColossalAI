import torch.nn as nn

from colossalai.legacy.context import ParallelMode, seed

from ..parallel_1d import *
from ..utils import get_tensor_parallel_mode
from ._utils import ColossalaiModule


class Dropout(ColossalaiModule):
    """Dropout layer of colossalai.

    Args:
        p (float, optional): probability of an element to be zeroed, defaults 0.5.
        inplace (bool, optional): whether to do dropout in-place, default to be False.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        tensor_parallel = get_tensor_parallel_mode()
        if tensor_parallel == "1d":
            drop = Dropout1D(p, inplace)
        else:
            drop = nn.Dropout(p, inplace)
        super().__init__(drop, tensor_parallel=tensor_parallel)

    def forward(self, *args):
        if self.tensor_parallel in [None, "1d"]:
            return super().forward(*args)
        else:
            with seed(ParallelMode.TENSOR):
                return super().forward(*args)
