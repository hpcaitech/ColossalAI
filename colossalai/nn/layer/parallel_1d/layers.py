#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Tuple

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.registry import LAYERS
from colossalai.utils import get_current_device
from .._common_utils import divide
from .._parallel_utilities import reduce_grad, reduce_input, gather_forward_split_backward, \
    split_forward_gather_backward
from ..base_layer import ParallelLayer


class Linear1D_Col(ParallelLayer):
    """Linear layer with column parallelism.

    The linear layer is defined as :math:`Y = XA + b`. A is parallelized along
    its second dimension as :math:`A = [A_1, ..., A_p]`.

    :param in_features: first dimension of matrix A.
    :type in_features: int
    :param output_size: second dimension of matrix A.
    :type output_size: int
    :param bias: If true, add bias, defaults to True
    :type bias: bool, optional
    :param dtype: The dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    :param gather_output: If true, call all-gether on output and make Y avaiable
                    to all GPUs, otherwise, every GPU will have its output
                    which is :math:`Y_i = XA_i`, defaults to False
    :type gather_output: bool, optional
    """

    def __init__(self,
                 in_features: int,
                 output_size: int,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 gather_output: bool = False):
        super().__init__()

        # Keep input parameters
        self.input_size = in_features
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = not bias

        world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
        self.output_size_per_partition = divide(output_size, world_size)

        # Parameters.
        # Initialize weight.
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}
        self.weight = Parameter(torch.empty(
            self.output_size_per_partition, self.input_size,
            **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(
                self.output_size_per_partition,
                **factory_kwargs))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        # Set up backprop all-reduce.
        input_parallel = reduce_grad(input_, ParallelMode.PARALLEL_1D)
        # Matrix multiply.

        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_forward_split_backward(
                output_parallel, ParallelMode.PARALLEL_1D, dim=-1)
        else:
            output = output_parallel
        if self.skip_bias_add:
            return output, self.bias
        else:
            return output


@LAYERS.register_module
class Linear1D_Row(ParallelLayer):
    """ Linear layer with row parallelism 

    :param in_features: size of each input sample
    :type in_features: int
    :param out_features: size of each output sample
    :type out_features: int
    :param bias: If set to ``False``, the layer will not learn an additive bias, defaults to True
    :type bias: bool, optional
    :param dtype: The dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    :param parallel_input: If set to ``False``, it's assumed that the input is splitted, defaults to False
    :type parallel_input: bool, optional
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 parallel_input: bool = False
                 ):
        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.parallel_input = parallel_input
        self.skip_bias_add = not bias

        # Divide the weight matrix along the last dimension.
        world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
        self.input_size_per_partition = divide(in_features, world_size)

        # Parameters.
        # Initialize weight.
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}
        self.weight = Parameter(torch.empty(
            self.out_features,
            self.input_size_per_partition,
            **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(
                self.out_features,
                **factory_kwargs
            ))

            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        init.xavier_normal_(self.weight)

    def forward(self, input_: Tensor) -> Tensor:
        # Set up backprop all-reduce.
        if self.parallel_input:
            input_ = input_
        else:
            input_ = split_forward_gather_backward(
                input_, ParallelMode.PARALLEL_1D, dim=-1)

        output_parallel = F.linear(input_, self.weight)
        output = reduce_input(output_parallel, ParallelMode.PARALLEL_1D)

        if not self.skip_bias_add:
            output = output + self.bias
        return output
