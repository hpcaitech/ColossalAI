import torch
import torch.nn as nn
from torch.optim import Optimizer

__all__ = ['Precision']


class Precision:

    def __init__(self, precision_type: torch.dtype, grad_clipping_type: str, grad_clipping_value: float):
        self.precision_type = precision_type
        self.grad_clipping_type = grad_clipping_type
        self.grad_clipping_value = grad_clipping_value

    def setup_model(self, model: nn.Module) -> nn.Module:
        # TODO: implement this method
        pass

    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        # TODO: implement this method
        # inject grad clipping and unscale loss
        pass

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        pass
