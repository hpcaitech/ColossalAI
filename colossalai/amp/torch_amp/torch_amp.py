#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn as nn
import torch.cuda.amp as torch_amp

from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from ._grad_scaler import GradScaler

from colossalai.nn.optimizer import ColossalaiOptimizer
from colossalai.utils import clip_grad_norm_fp32


class TorchAMPOptimizer(ColossalaiOptimizer):
    """A wrapper class which integrate pytorch amp with an optimizer

    :param optim: a normal optimizer like Adam or SGD
    :type optim: torch.optim.Optimizer
    :param init_scale: Initial scale factor
    :type init_scale: float, optional, default=2.**16
    :param growth_factor: Factor by which the scale is multiplied during :meth:`update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
    :type growth_factor: float, optional, default=2.0
    :param backoff_factor: Factor by which the scale is multiplied during :meth:`update` if inf/NaN gradients occur in an iteration.
    :type backoff_factor: float, optional, default=0.5
    :param growth_interval: Number of consecutive iterations without inf/NaN gradients that must occur for the scale to be multiplied by ``growth_factor``.
    :type growth_interval: int, optional, default=2000
    :param enabled: If ``False``, disables gradient scaling. :meth:`step` simply invokes the underlying ``optimizer.step()``, and other methods become no-ops.
    :type enabled: bool, optional, default=True

    """

    def __init__(self, optim: Optimizer, *args, **kwargs):
        super().__init__(optim)
        self.scaler = GradScaler(*args, **kwargs)

    def backward(self, loss: Tensor):
        """backward with torch amp gradient scaler
        :param loss: loss computed by a loss function
        :type loss: torch.Tensor
        """
        self.scaler.scale(loss).backward()

    def step(self):
        """update the parameters of the model
        """
        self.scaler.step(self.optim)
        self.scaler.update()

    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        """apply gradient clipping to the model parameters
        :param model: your model object
        :type model: torch.nn.Module
        :param max_norm: max norm value for gradient clipping
        :type max_norm: float
        """
        if max_norm > 0.0:
            self.scaler.unscale_(self.optim)
            clip_grad_norm_fp32(model.parameters(), max_norm)


class TorchAMPModel(nn.Module):
    """A wrapper class for a model object which executes forward with values automatically
    cast to fp16
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @torch_amp.autocast()
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class TorchAMPLoss(nn.Module):
    """A wrapper class for a criterion object which computes the loss in mixed-precision context
    :param loss: a loss function object
    :type loss: torch.nn.modules.loss._Loss
    """

    def __init__(self, loss: _Loss):
        super().__init__()
        self.loss = loss

    @torch_amp.autocast()
    def forward(self, *args, **kwargs):
        return self.loss(*args, **kwargs)
