#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import collections.abc
from itertools import repeat

import numpy as np
import torch
from colossalai.constants import IS_TENSOR_PARALLEL, NUM_PARTITIONS
from colossalai.utils import checkpoint
from torch import Tensor, nn


class CheckpointModule(nn.Module):
    def __init__(self, checkpoint: bool = True):
        super().__init__()
        self.checkpoint = checkpoint
        self._use_checkpoint = checkpoint

    def _forward(self, *args, **kwargs):
        raise NotImplementedError('CheckpointModule should implement _forward method instead of origin forward')

    def forward(self, *args, **kwargs):
        if self._use_checkpoint:
            return checkpoint(self._forward, *args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def train(self, mode: bool = True):
        self._use_checkpoint = self.checkpoint
        return super().train(mode=mode)

    def eval(self):
        self._use_checkpoint = False
        return super().eval()


def divide(numerator, denominator):
    """ only allow exact division """
    assert numerator % denominator == 0, \
        '{} is not divisible by {}'.format(numerator, denominator)
    return numerator // denominator


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


def set_tensor_parallel_attribute_by_size(param, size):
    setattr(param, IS_TENSOR_PARALLEL, True)
    setattr(param, NUM_PARTITIONS, size // np.prod(param.shape))


def set_tensor_parallel_attribute_by_partition(param, num_partitions):
    setattr(param, IS_TENSOR_PARALLEL, True)
    setattr(param, NUM_PARTITIONS, num_partitions)


# From PyTorch internals


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
