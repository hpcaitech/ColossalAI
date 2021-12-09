#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.constants import (INPUT_GROUP_3D, OUTPUT_GROUP_3D,
                                  WEIGHT_GROUP_3D)
from colossalai.context import ParallelMode, seed
from colossalai.core import global_context as gpc
from colossalai.nn.init import init_bias_, init_weight_
from colossalai.registry import LAYERS
from colossalai.utils import get_current_device
from torch import Tensor, dtype
from torch.nn import Parameter
from torch.nn import init as init

from .._common_utils import divide, set_tensor_parallel_attribute_by_size
from ._operation import (Add_3D, Matmul_AB_3D, Mul_3D, Sum_3D, layer_norm_3d,
                         linear_3d)
from ._utils import (get_depth_from_env, get_last_group,
                     get_parallel_mode_from_env, swap_in_out_group)


@LAYERS.register_module
class LayerNorm3D(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        # input_parallel_mode: ParallelMode,
        # weight_parallel_mode: ParallelMode,
        eps: float = 1e-12,
        dtype: dtype = None,
    ):
        super().__init__()
        self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        self.output_parallel_mode = get_last_group(self.input_parallel_mode,
                                                   self.weight_parallel_mode)
        self.depth = get_depth_from_env()
        self.normalized_shape = normalized_shape
        self.normalized_shape_per_partition = divide(normalized_shape, self.depth)

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
        set_tensor_parallel_attribute_by_size(self.weight, self.normalized_shape)
        set_tensor_parallel_attribute_by_size(self.bias, self.normalized_shape)

    def reset_parameters(self):
        init.zeros_(self.bias)
        init.ones_(self.weight)

    def forward(self, input_: Tensor) -> Tensor:
        # '''x = weight * (x - mean) / sqrt(var + eps) + bias'''
        # # input: [m/q^2, n, h/q]
        # # [m/q^2, n, 1]
        # mean = Sum_3D.apply(input_, -1, self.depth, self.output_parallel_mode,
        #                     True) / self.normalized_shape
        # # [m/q^2, n, 1]
        # var = (input_ - mean).pow(2)
        # var = Sum_3D.apply(var, -1, self.depth, self.output_parallel_mode,
        #                    True) / self.normalized_shape

        # output = (input_ - mean) / torch.sqrt(var + self.variance_epsilon)
        # output = Mul_3D.apply(output, self.weight, self.depth,
        #                       self.input_parallel_mode,
        #                       self.weight_parallel_mode,
        #                       self.output_parallel_mode)
        # output = Add_3D.apply(output, self.bias, self.depth,
        #                       self.input_parallel_mode,
        #                       self.weight_parallel_mode,
        #                       self.output_parallel_mode)
        # return output
        return layer_norm_3d.apply(input_, self.weight, self.bias,
                                   self.normalized_shape,
                                   self.variance_epsilon,
                                   self.input_parallel_mode,
                                   self.weight_parallel_mode,
                                   self.output_parallel_mode)

    def extra_repr(self):
        return '{}, eps={}'.format(self.normalized_shape,
                                   self.variance_epsilon)


@LAYERS.register_module
class Linear3D(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            #  input_parallel_mode: ParallelMode,
            #  weight_parallel_mode: ParallelMode,
            bias: bool = True,
            dtype: dtype = None,
            init_weight: str = 'torch',
            init_bias: str = 'torch'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        self.output_parallel_mode = get_last_group(self.input_parallel_mode,
                                                   self.weight_parallel_mode)
        # self.with_bias = bias
        self.depth = get_depth_from_env()
        self.in_features_per_partition = divide(in_features, self.depth)
        self.out_features_per_partition = divide(out_features, self.depth)

        # [k/q, h/q]
        self.weight = Parameter(
            torch.empty(self.in_features_per_partition,
                        self.out_features_per_partition,
                        device=get_current_device(),
                        dtype=dtype))

        # [h/q]
        if bias:
            self.bias = Parameter(
                torch.zeros(self.out_features_per_partition,
                            device=get_current_device(),
                            dtype=dtype))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(init_weight, init_bias)
        self._set_tensor_parallel_attributes()
        swap_in_out_group()

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_size(self.weight, self.in_features * self.out_features)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_size(self.bias, self.out_features)

    def reset_parameters(self, init_weight, init_bias) -> None:
        # setting
        fan_in, fan_out = self.in_features, self.out_features
        weight_src_rank = gpc.get_ranks_in_group(self.weight_parallel_mode)[0]
        output_src_rank = gpc.get_ranks_in_group(self.output_parallel_mode)[0]

        # init weight
        init_weight_(self.weight, fan_in, fan_out, init_method=init_weight)
        dist.broadcast(self.weight,
                       src=weight_src_rank,
                       group=gpc.get_group(self.weight_parallel_mode))
        # init bias
        if self.bias is not None:
            init_bias_(self.bias, fan_in, init_method=init_bias)
            dist.broadcast(self.bias,
                           src=weight_src_rank,
                           group=gpc.get_group(self.weight_parallel_mode))
            dist.broadcast(self.bias,
                           src=output_src_rank,
                           group=gpc.get_group(self.output_parallel_mode))

    def forward(self, input_: Tensor) -> Tensor:
        # # input: [m/q^2, n, k/q]
        # # output: [m/q^2, n, h/q]
        # output = Matmul_AB_3D.apply(input_, self.weight, self.depth,
        #                             self.input_parallel_mode,
        #                             self.weight_parallel_mode,
        #                             self.output_parallel_mode)

        # if self.bias is not None:
        #     output = Add_3D.apply(output, self.bias, self.depth,
        #                           self.output_parallel_mode,
        #                           self.weight_parallel_mode,
        #                           self.input_parallel_mode)
        # return output
        return linear_3d.apply(input_, self.weight, self.bias,
                               self.input_parallel_mode,
                               self.weight_parallel_mode,
                               self.output_parallel_mode)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.with_bias)
