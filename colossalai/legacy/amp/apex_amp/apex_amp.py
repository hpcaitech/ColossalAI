#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn as nn

try:
    import apex.amp as apex_amp
except ImportError:
    pass

from torch import Tensor

from colossalai.interface import OptimizerWrapper
from colossalai.legacy.utils import clip_grad_norm_fp32


class ApexAMPOptimizer(OptimizerWrapper):
    """A wrapper class for APEX optimizer and it implements apex-specific backward and clip_grad_norm
    methods
    """

    def backward(self, loss: Tensor):
        """Backward pass to get all gradients

        Args:
            loss (torch.Tensor): Loss computed by a loss function
        """
        with apex_amp.scale_loss(loss, self.optim) as scaled_loss:
            scaled_loss.backward()

    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        """Clip gradients by norm

        Args:
            model (torch.nn.Module): Your model object
            max_norm (float): The max norm value for gradient clipping
        """
        if max_norm > 0:
            clip_grad_norm_fp32(apex_amp.master_params(self.optim), max_norm)
