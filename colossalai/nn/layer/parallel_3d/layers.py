#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from typing import Tuple

import torch
import torch.nn as nn
from colossalai.context import ParallelMode, seed
from colossalai.registry import LAYERS
from colossalai.utils import get_current_device
from torch import Tensor, dtype
from torch.nn import Parameter

from .._common_utils import divide, set_tensor_parallel_attribute
from ._operation import Add_3D, Matmul_AB_3D, Mul_3D, Sum_3D
from ._utils import get_depth_from_env, get_last_group


@LAYERS.register_module
class LayerNorm3D(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        eps: float = 1e-12,
        dtype: dtype = None,
    ):
        super().__init__()
        self.input_parallel_mode = input_parallel_mode
        self.weight_parallel_mode = weight_parallel_mode
        self.output_parallel_mode = get_last_group(self.input_parallel_mode,
                                                   self.weight_parallel_mode)
        self.depth = get_depth_from_env()
        self.normalized_shape = normalized_shape
        self.normalized_shape_per_partition = divide(normalized_shape,
                                                     self.depth**2)

        self.weight = Parameter(
            torch.ones(self.normalized_shape_per_partition,
                       device=get_current_device(),
                       dtype=dtype))
        self.bias = Parameter(
            torch.zeros(self.normalized_shape_per_partition,
                        device=get_current_device(),
                        dtype=dtype))
        self.variance_epsilon = eps
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute(self.weight)
        set_tensor_parallel_attribute(self.bias)

    def groups_for_next_layer(self) -> Tuple[ParallelMode, ParallelMode]:
        return self.input_parallel_mode, self.weight_parallel_mode

    def reset_parameters(self):
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.weight)

    def forward(self, input_: Tensor) -> Tensor:
        '''x = weight * (x - mean) / sqrt(var + eps) + bias'''
        # input: [m/q^2, n, h/q]
        # [m/q^2, n, 1]
        mean = Sum_3D.apply(input_, -1, self.depth, self.output_parallel_mode,
                            True) / self.normalized_shape
        # [m/q^2, n, 1]
        var = (input_ - mean).pow(2)
        var = Sum_3D.apply(var, -1, self.depth, self.output_parallel_mode,
                           True) / self.normalized_shape

        output = (input_ - mean) / torch.sqrt(var + self.variance_epsilon)
        output = Mul_3D.apply(output, self.weight, self.depth,
                              self.input_parallel_mode,
                              self.weight_parallel_mode,
                              self.output_parallel_mode)
        output = Add_3D.apply(output, self.bias, self.depth,
                              self.input_parallel_mode,
                              self.weight_parallel_mode,
                              self.output_parallel_mode)
        return output

    def extra_repr(self):
        return '{}, eps={}'.format(self.normalized_shape,
                                   self.variance_epsilon)


@LAYERS.register_module
class Linear3D(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 input_parallel_mode: ParallelMode,
                 weight_parallel_mode: ParallelMode,
                 bias: bool = True,
                 dtype: dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_parallel_mode = input_parallel_mode
        self.weight_parallel_mode = weight_parallel_mode
        self.output_parallel_mode = get_last_group(self.input_parallel_mode,
                                                   self.weight_parallel_mode)
        self.with_bias = bias
        self.depth = get_depth_from_env()
        self.in_features_per_partition = divide(in_features, self.depth)
        self.out_features_per_partition = divide(out_features, self.depth**2)

        # [k/q, h/q^2]
        self.weight = Parameter(
            torch.empty(self.in_features_per_partition,
                        self.out_features_per_partition,
                        device=get_current_device(),
                        dtype=dtype))

        # [h/q^2]
        if bias:
            self.bias = Parameter(
                torch.zeros(self.out_features_per_partition,
                            device=get_current_device(),
                            dtype=dtype))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute(self.weight)
        if self.bias is not None:
            set_tensor_parallel_attribute(self.bias)

    def groups_for_next_layer(self) -> Tuple[ParallelMode, ParallelMode]:
        return self.output_parallel_mode, self.weight_parallel_mode

    def reset_parameters(self):
        # setting
        fan_in = self.in_features
        a = math.sqrt(5)
        nonlinearity = 'leaky_relu'

        # init weight
        std = nn.init.calculate_gain(nonlinearity, a) / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std
        with seed(ParallelMode.TENSOR):
            nn.init.uniform_(self.weight, -bound, bound)

        # init bias
        if self.with_bias:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            with seed(ParallelMode.TENSOR):
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: Tensor) -> Tensor:
        # input: [m/q^2, n, k/q]
        # output: [m/q^2, n, h/q]
        output = Matmul_AB_3D.apply(input_, self.weight, self.depth,
                                    self.input_parallel_mode,
                                    self.weight_parallel_mode,
                                    self.output_parallel_mode)

        if self.with_bias:
            output = Add_3D.apply(output, self.bias, self.depth,
                                  self.output_parallel_mode,
                                  self.weight_parallel_mode,
                                  self.input_parallel_mode)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.with_bias)
