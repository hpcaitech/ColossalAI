#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math

import numpy as np
from colossalai.utils.common import print_rank_0
import torch
from colossalai.constants import IS_TENSOR_PARALLEL, NUM_PARTITIONS
from colossalai.utils import checkpoint
from torch import Tensor, nn


def divide(numerator, denominator):
    """ only allow exact division """
    assert numerator % denominator == 0, \
        '{} is not divisible by {}'.format(numerator, denominator)
    return numerator // denominator


def gelu(x: Tensor) -> Tensor:
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


def set_tensor_parallel_attribute(param, size):
    # if not hasattr(param, IS_TENSOR_PARALLEL):
    setattr(param, IS_TENSOR_PARALLEL, True)
    # if not hasattr(param, NUM_PARTITIONS):
    setattr(param, NUM_PARTITIONS, size // np.prod(param.shape))


class CheckpointModule(nn.Module):
    def __init__(self, checkpoint: bool = True):
        super().__init__()
        self.checkpoint = checkpoint
        self._use_checkpoint = checkpoint

    def _forward(self, *args):
        raise NotImplementedError(
            'CheckpointModule should implement _forward method instead of origin forward')

    def forward(self, *args):
        if self._use_checkpoint:
            return checkpoint(self._forward, *args)
        else:
            return self._forward(*args)

    def train(self, mode: bool = True):
        self._use_checkpoint = self.checkpoint
        return super().train(mode=mode)

    def eval(self):
        self._use_checkpoint = False
        return super().eval()
