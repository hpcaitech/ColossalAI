#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import collections.abc
from itertools import repeat

import numpy as np
import torch
from torch import Tensor, nn

from colossalai.legacy.constants import IS_TENSOR_PARALLEL, NUM_PARTITIONS
from colossalai.legacy.global_variables import tensor_parallel_env as env
from colossalai.legacy.utils import checkpoint


class CheckpointModule(nn.Module):
    def __init__(self, checkpoint: bool = True, offload: bool = False):
        super().__init__()
        self.checkpoint = checkpoint
        self._use_checkpoint = checkpoint
        self._offload = offload

    def _forward(self, *args, **kwargs):
        raise NotImplementedError("CheckpointModule should implement _forward method instead of origin forward")

    def forward(self, *args, **kwargs):
        if self._use_checkpoint:
            return checkpoint(self._forward, self._offload, *args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def train(self, mode: bool = True):
        self._use_checkpoint = self.checkpoint
        return super().train(mode=mode)

    def eval(self):
        self._use_checkpoint = False
        return super().eval()


def divide(numerator, denominator):
    """Only allow exact division.

    Args:
        numerator (int): Numerator of the division.
        denominator (int): Denominator of the division.

    Returns:
        int: the result of exact division.
    """
    assert denominator != 0, "denominator can not be zero"
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)
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


def get_tensor_parallel_mode():
    return env.mode


# From PyTorch internals


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
