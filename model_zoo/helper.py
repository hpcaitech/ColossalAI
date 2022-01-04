import torch
import torch.nn as nn
from colossalai.nn.layer import WrappedDropPath as DropPath


class TransformerLayer(nn.Module):
    """Transformer layer builder.
    """
    def __init__(self,
                 att: nn.Module,
                 ffn: nn.Module,
                 norm1: nn.Module,
                 norm2: nn.Module,
                 droppath=None,
                 droppath_rate: float = 0):
        super().__init__()
        self.att = att
        self.ffn = ffn
        self.norm1 = norm1
        self.norm2 = norm2
        self.droppath = DropPath(droppath_rate) if droppath is None else droppath

    def forward(self, x):
        x = x + self.droppath(self.att(self.norm1(x)))
        x = x + self.droppath(self.ffn(self.norm2(x)))
        return x
