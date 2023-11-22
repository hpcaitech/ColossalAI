#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from colossalai.utils.device import autocast

import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from colossalai.interface import OptimizerWrapper
from colossalai.legacy.utils import clip_grad_norm_fp32

from ._grad_scaler import GradScaler


class TorchAMPOptimizer(OptimizerWrapper):
    """A wrapper class which integrate Pytorch AMP with an optimizer

    Args:
        optim (torch.optim.Optimizer): A normal optimizer like Adam or SGD.
        init_scale (float, optional, default=2.**16):  Initial scale factor.
        growth_factor (float, optional, default=2.0):  Factor by which the scale is multiplied during
            :meth:`update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
        backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
            :meth:`update` if inf/NaN gradients occur in an iteration.
        growth_interval (int, optional, default=2000):  Number of consecutive iterations without inf/NaN gradients
            that must occur for the scale to be multiplied by ``growth_factor``.
        enabled (bool, optional, default=True):  If ``False``, disables gradient scaling. :meth:`step` simply
            invokes the underlying ``optimizer.step()``, and other methods become no-ops.
    """

    def __init__(self, optim: Optimizer, *args, **kwargs):
        super().__init__(optim)
        self.scaler = GradScaler(*args, **kwargs)

    def backward(self, loss: Tensor):
        """Backward with torch amp gradient scaler

        Args:
            loss (torch.Tensor): Loss computed by a loss function
        """
        self.scaler.scale(loss).backward()

    def step(self):
        """Update the parameters of the model"""
        self.scaler.step(self.optim)
        self.scaler.update()

    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        """Apply gradient clipping to the model parameters

        Args:
            model (torch.nn.Module): Your model object
            max_norm (float): Max norm value for gradient clipping
        """
        if max_norm > 0.0:
            self.scaler.unscale_(self.optim)
            clip_grad_norm_fp32(model.parameters(), max_norm)


class TorchAMPModel(nn.Module):
    """A wrapper class for a model object which executes forward with values automatically
    cast to fp16

    Args:
        model (:class:`torch.nn.Module`): a torch model instance
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @autocast()
    def forward(self, *args, **kwargs):
        """
        Execute forward under the torch amp context
        """
        return self.model(*args, **kwargs)


class TorchAMPLoss(nn.Module):
    """A wrapper class for a criterion object which computes the loss in mixed-precision context

    Args:
        loss (torch.nn.modules.loss._Loss): A loss function object
    """

    def __init__(self, loss: _Loss):
        super().__init__()
        self.loss = loss

    @autocast()
    def forward(self, *args, **kwargs):
        """
        Execute forward under the torch amp context
        """
        return self.loss(*args, **kwargs)
