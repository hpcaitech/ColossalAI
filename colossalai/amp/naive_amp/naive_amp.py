#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, List, Any, Dict
from torch.optim import Optimizer
import torch.cuda.amp as torch_amp

from colossalai.nn.optimizer import ColossalaiOptimizer
from ._fp16_optimizer import FP16Optimizer


class NaiveAMPOptimizer(ColossalaiOptimizer):
    """A wrapper class for optimizer to cast all parameters to fp16

    :param optim: a normal optimizer like Adam or SGD
    :type optim: torch.optim.Optimizer
    """

    def __init__(self, optim: Optimizer, *args, **kwargs):
        optim = FP16Optimizer(optimizer=optim, *args, **kwargs)
        super().__init__(optim)

    def backward(self, loss: Tensor):
        """backward with gradient scaler
        :param loss: loss computed by a loss function
        :type loss: torch.Tensor
        """
        loss = self.optim.scale_loss(loss)
        loss.backward()

    def step(self):
        return self.optim.step()

    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        pass


class NaiveAMPModel(nn.Module):
    """A wrapper class for model to cast the model into fp16 and 
    automatically cast the input and output
    """

    def __init__(self,
                 model: nn.Module,
                 output_to_fp32: bool = True):
        super().__init__()
        self.model = model.half()
        self._output_to_fp32 = output_to_fp32

    def _convert_to_fp16(self, input_: Any):
        if isinstance(input_, Tensor) and input_.dtype == torch.float32:
            input_ = input_.half()
        return input_

    def _convert_to_fp32(self, input_: Any):
        if isinstance(input_, Tensor) and input_.dtype == torch.float16:
            input_ = input_.float()
        return input_

    def forward(self, *args, **kwargs):
        if args:
            args = [self._convert_to_fp16(arg) for arg in args]
        if kwargs:
            for k, v in kwargs.items():
                kwargs[k] = self._convert_to_fp16(v)

        out = self.model(*args, **kwargs)

        if self._output_to_fp32:
            if isinstance(out, Tensor):
                out = self._convert_to_fp32(out)
            elif isinstance(out, (tuple, list)):
                out = [self._convert_to_fp32(val) for val in out]
        return out
