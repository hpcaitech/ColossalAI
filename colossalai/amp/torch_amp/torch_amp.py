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

    def __init__(self, optim: Optimizer, *args, **kwargs):
        super().__init__(optim)
        self.scaler = GradScaler(*args, **kwargs)

    def backward(self, loss: Tensor):
        self.scaler.scale(loss).backward()

    def step(self):
        self.scaler.step(self.optim)
        self.scaler.update()

    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        if max_norm > 0.0:
            self.scaler.unscale_(self.optim)
            clip_grad_norm_fp32(model.parameters(), max_norm)


class TorchAMPModel(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @torch_amp.autocast()
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class TorchAMPLoss(nn.Module):

    def __init__(self, loss: _Loss):
        super().__init__()
        self.loss = loss

    @torch_amp.autocast()
    def forward(self, *args, **kwargs):
        return self.loss(*args, **kwargs)
