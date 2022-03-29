from colossalai.utils import get_current_device
from torch import nn
from colossalai import kernel

from ... import init as init
from ..parallel_1d import *
from ..parallel_2d import *
from ..parallel_2p5d import *
from ..parallel_3d import *
from ..utils import get_tensor_parallel_mode
from ..vanilla import *

_parallel_layernorm = {
    '1d': kernel.LayerNorm,
    '2d': LayerNorm2D,
    '2.5d': LayerNorm2p5D,
    '3d': LayerNorm3D
}


class LayerNorm(nn.Module):
    r"""Layer Normalization for colossalai.

    Args:
        normalized_shape (int): input shape from an expected input of size.
            :math:`[* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1] \times \ldots \times \text{normalized_shape}[-1]]`
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps (float, optional): a value added to the denominator for numerical stability, defaults to 1e-05
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
    """

    def __init__(self, normalized_shape: int, eps=1e-05, dtype=None) -> None:
        super().__init__()
        tensor_parallel = get_tensor_parallel_mode()
        if tensor_parallel is None:
            self.norm = nn.LayerNorm(normalized_shape, eps=eps).to(dtype).to(get_current_device())
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
