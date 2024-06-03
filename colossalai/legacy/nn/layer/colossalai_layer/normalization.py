from torch import nn

from colossalai.accelerator import get_accelerator

from ..parallel_1d import LayerNorm1D
from ..parallel_2d import LayerNorm2D
from ..parallel_2p5d import LayerNorm2p5D
from ..parallel_3d import LayerNorm3D
from ..utils import get_tensor_parallel_mode
from ..vanilla import VanillaLayerNorm
from ._utils import ColossalaiModule

_parallel_layernorm = {
    None: VanillaLayerNorm,
    "1d": LayerNorm1D,
    "2d": LayerNorm2D,
    "2.5d": LayerNorm2p5D,
    "3d": LayerNorm3D,
}


class LayerNorm(ColossalaiModule):
    r"""Layer Normalization for colossalai.

    Args:
        normalized_shape (int): input shape from an expected input of size.
            :math:`[* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1]
            \times \ldots \times \text{normalized_shape}[-1]]`
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps (float): a value added to the denominator for numerical stability, defaults to 1e-05.
        bias (bool, optional): Whether to add a bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
    """

    def __init__(self, normalized_shape: int, eps=1e-05, bias=True, dtype=None) -> None:
        tensor_parallel = get_tensor_parallel_mode()
        if tensor_parallel is None:
            norm = nn.LayerNorm(normalized_shape, eps=eps).to(dtype).to(get_accelerator().get_current_device())
        else:
            norm = _parallel_layernorm[tensor_parallel](normalized_shape, eps=eps, dtype=dtype)
        super().__init__(norm)
