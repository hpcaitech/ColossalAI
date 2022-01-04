import torch
import torch.nn as nn
from colossalai.nn.layer import WrappedDropPath as DropPath


class TransformerLayer(nn.Module):
    """Transformer layer builder.
    """
    def __init__(self,
                 SA: nn.Module,
                 FFN: nn.Module,
                 NORM1: nn.Module,
                 NORM2: nn.Module,
                 DROPPATH=None,
                 droppath_rate: float = 0):
        super().__init__()
        self.SA = SA
        self.FFN = FFN
        self.NORM1 = NORM1
        self.NORM2 = NORM2
        self.DROPPATH = DropPath(droppath_rate) if DROPPATH is None else DROPPATH

    def forward(self, x):
        x = x + self.DROPPATH(self.SA(self.NORM1(x)))
        x = x + self.DROPPATH(self.FFN(self.NORM2(x)))
        return x
