#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn as nn
try:
    import apex.amp as apex_amp
except:
    pass
from torch import Tensor

from colossalai.nn.optimizer import ColossalaiOptimizer
from colossalai.utils import clip_grad_norm_fp32


class ApexAMPOptimizer(ColossalaiOptimizer):
    """ A wrapper class for APEX optimizer and it implements apex-specific backward and clip_grad_norm
    methods
    """

    def backward(self, loss: Tensor):
        """Backward pass to get all gradients

        :param loss: Loss computed by a loss function
        :type loss: torch.Tensor
        """
        with apex_amp.scale_loss(loss, self.optim) as scaled_loss:
            scaled_loss.backward()

    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        """Clip gradients' norm

        :param model: Your model object
        :type model: torch.nn.Module
        :param max_norm: The max norm value for gradient clipping
        :type max_norm: float
        """
        if max_norm > 0:
            clip_grad_norm_fp32(apex_amp.master_params(self.optim), max_norm)
