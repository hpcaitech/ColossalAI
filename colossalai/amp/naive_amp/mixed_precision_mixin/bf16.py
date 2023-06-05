import torch
from torch import Tensor

from .base import MixedPrecisionMixin


class BF16MixedPrecisionMixin(MixedPrecisionMixin):
    dtype = torch.bfloat16

    def pre_backward(self, loss: Tensor) -> Tensor:
        return loss

    def pre_backward_by_grad(self, tensor: Tensor, grad: Tensor) -> Tensor:
        return grad

    def should_skip_step(self) -> bool:
        return False

    def pre_zero_grad(self) -> None:
        pass

    def get_grad_div_scale(self) -> float:
        return 1.0
