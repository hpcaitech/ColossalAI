from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.utils.device import autocast

from .mixed_precision_base import MixedPrecision

__all__ = ["FP16_Torch_MixedPrecision", "TorchAMPOptimizer", "TorchAMPModule"]


class TorchAMPOptimizer(OptimizerWrapper):
    """
    Optimizer wrapper for mixed precision training in FP16 using PyTorch AMP.

    Args:
        optim (Optimizer): Optimizer to wrap.
        init_scale (float): Initial scale factor. Default: 2**16.
        growth_factor (float): Factor by which the scale is multiplied during
            :meth:`torch.cuda.amp.GradScaler.step` if gradients were found to be finite
            this iteration. Default: 2.0.
        backoff_factor (float): Factor by which the scale is multiplied during
            :meth:`torch.cuda.amp.GradScaler.step` if gradients were found to be infinite
            this iteration. Default: 0.5.
        growth_interval (int): Number of iterations between :meth:`torch.cuda.amp.GradScaler.step`
            calls that may cause the scale to increase. Default: 2000.
    """

    def __init__(
        self,
        optim: Optimizer,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ) -> None:
        super().__init__(optim)
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
        )

    def backward(self, loss: Tensor, *args, **kwargs) -> None:
        scaled_loss = self.scale_loss(loss)
        scaled_loss.backward(*args, **kwargs)

    def step(self, *args, **kwargs) -> Optional[float]:
        out = self.scaler.step(self.optim, *args, **kwargs)
        self.scaler.update()
        return out

    def scale_loss(self, loss: Tensor) -> Tensor:
        return self.scaler.scale(loss)

    def unscale_grad(self) -> None:
        self.scaler.unscale_(self.optim)

    def clip_grad_by_value(self, clip_value: float, *args, **kwargs) -> None:
        self.unscale_grad()
        super().clip_grad_by_value(clip_value, *args, **kwargs)

    def clip_grad_by_norm(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self.unscale_grad()
        super().clip_grad_by_norm(max_norm, norm_type, error_if_nonfinite, *args, **kwargs)


class TorchAMPModule(ModelWrapper):
    """
    Module wrapper for mixed precision training in FP16 using PyTorch AMP.

    Args:
        module (nn.Module): Module to wrap.
    """

    def __init__(self, module: nn.Module):
        super().__init__(module)

    def forward(self, *args, **kwargs):
        with autocast():
            return self.module(*args, **kwargs)


class FP16TorchMixedPrecision(MixedPrecision):
    """
    Precision for mixed precision training in FP16 using PyTorch AMP.

    Args:
        init_scale (float): Initial scale factor. Default: 2**16.
        growth_factor (float): Factor by which the scale is multiplied during
            :meth:`torch.cuda.amp.GradScaler.step` if gradients were found to be finite
            this iteration. Default: 2.0.
        backoff_factor (float): Factor by which the scale is multiplied during
            :meth:`torch.cuda.amp.GradScaler.step` if gradients were found to be infinite
            this iteration. Default: 0.5.
        growth_interval (int): Number of iterations between :meth:`torch.cuda.amp.GradScaler.step`
            calls that may cause the scale to increase. Default: 2000.
    """

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ) -> None:
        super().__init__()
        self.torch_amp_kwargs = dict(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
        )

    def configure(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
    ) -> Tuple[nn.Module, OptimizerWrapper, Callable]:
        model = TorchAMPModule(model)
        if optimizer is not None:
            optimizer = TorchAMPOptimizer(optimizer, **self.torch_amp_kwargs)
        if criterion is not None:
            criterion = TorchAMPModule(criterion)
        return model, optimizer, criterion
