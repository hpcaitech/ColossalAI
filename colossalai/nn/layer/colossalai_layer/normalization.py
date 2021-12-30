from typing import Optional

from colossalai.utils import get_current_device
from torch import nn

from ... import init as init
from ..parallel_1d import *
from ..parallel_2d import *
from ..parallel_2p5d import *
from ..parallel_3d import *
from ..utils import get_tensor_parallel_mode
from ..vanilla import *

_parallel_layernorm = {'2d': LayerNorm2D, '2.5d': LayerNorm2p5D, '3d': LayerNorm3D}


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps=1e-05, dtype=None) -> None:
        super().__init__()
        tensor_parallel = get_tensor_parallel_mode()
        if tensor_parallel in ['None', '1d']:
            self.norm = nn.LayerNorm(normalized_shape, eps=eps, device=get_current_device(), dtype=dtype)
        else:
            self.norm = _parallel_layernorm[tensor_parallel](normalized_shape, eps=eps, dtype=dtype)

    @property
    def weight(self):
        return self.norm.weight

    @property
    def bias(self):
        return self.norm.bias

    def forward(self, *args):
        return self.norm(*args)
