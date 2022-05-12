from .colo_tensor import ColoTensor
from .const import TensorType
import torch


class ColoParameter(ColoTensor):
    r"""A kind of ColoTensor to be considered as a module parameter.

    """

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self._type = TensorType.MODEL

    def __new__(cls, *args, **kwargs):
        t = super(ColoParameter, cls).__new__(cls)
        t._type = TensorType.MODEL
        return t

    @staticmethod
    def init_from_torch_tensor(tensor: torch.Tensor, save_payload=True) -> 'ColoParameter':
        colo_p = ColoParameter(*tensor.size(),
                               dtype=tensor.dtype,
                               requires_grad=tensor.requires_grad,
                               pin_memory=tensor.is_pinned(),
                               device=tensor.device,
                               torch_tensor=tensor if save_payload else torch.empty(0))
        return colo_p
