#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from typing import Callable, List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter

from colossalai.nn import init as init
from colossalai.nn.layer.utils import divide
from colossalai.tensor.d_tensor import shard_colwise, shard_rowwise, sharded_tensor_to_param

from ._operation import (
    gather_forward_split_backward,
    linear_with_async_comm,
    reduce_forward,
    split_forward_gather_backward,
)
from .parallel_module import ParallelModule
from .utils import create_randomizer_with_offset

__all__ = ['Linear1D_Col', 'Linear1D_Row']


class Linear1D_Col(ParallelModule):
    r"""Linear layer with column parallelism.

    The linear layer is defined as :math:`Y = XA + b`. A is parallelized along
    its second dimension as :math:`A = [A_1, ..., A_p]`.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (`torch.dtype`): The dtype of parameters, defaults to None.
        device (`torch.device`): The device of parameters, defaults to None.
        process_group (`torch.distributed.ProcessGroup`): The process group to be used for weight sharding and communication, defaults to None.
        gather_output (bool, optional): If true, call all-gather on output and make Y available
                    to all GPUs, otherwise, every GPU will have its output
                    which is :math:`Y_i = XA_i`, defaults to False
        skip_bias_add (bool): If set to ``True``, it will skip bias add for linear layer,
            which is preserved for kernel fusion, defaults to False
        weight_initializer (`typing.Callable`):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (`typing.Callable`):
            The initializer of bias, defaults to xavier uniform initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 process_group: ProcessGroup = None,
                 gather_output: bool = False,
                 skip_bias_add: bool = False,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)):
        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.device = device
        self.process_group = process_group

        if skip_bias_add and not bias:
            raise ValueError('cannot skip bias addition if bias is None')

        # Parameters.
        factory_kwargs = {'device': device, 'dtype': dtype}

        weight = torch.empty(self.out_features, self.in_features, **factory_kwargs)
        sharded_weight = shard_rowwise(weight, self.process_group)
        self.weight = sharded_tensor_to_param(sharded_weight)

        if bias:
            bias = torch.empty(self.out_features, **factory_kwargs)
            sharded_bias = shard_colwise(bias, self.process_group)
            self.bias = sharded_tensor_to_param(sharded_bias)
        else:
            self.bias = None

        # offset the seed with randomizer index and rank
        seed = torch.random.initial_seed()
        self.randomizer = create_randomizer_with_offset(seed, process_group=self.process_group)

        # init weights
        self.reset_parameters(weight_initializer, bias_initializer)

    @staticmethod
    def from_native_module(module: nn.Linear, process_group: Union[ProcessGroup, List[ProcessGroup]], *args,
                           **kwargs) -> ParallelModule:
        r"""
        Convert a native PyTorch linear layer to a parallelized linear layer.
        """
        # get the attributes
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        device = module.weight.device

        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, \
                f'Expected only one process group, got {len(process_group)}.'
            process_group = process_group[0]

        linear_1d = Linear1D_Col(in_features=in_features,
                                 out_features=out_features,
                                 bias=bias,
                                 device=device,
                                 process_group=process_group,
                                 *args,
                                 **kwargs)

        with torch.no_grad():
            # the weigh to the linear layer is a transpose
            # thus shard on row is equal to shard on column
            sharded_weight = shard_rowwise(module.weight.data, process_group)
            linear_1d.weight.data.copy_(sharded_weight)
            if bias:
                sharded_bias = shard_colwise(module.bias.data, process_group)
                linear_1d.bias.copy_(sharded_bias)
        return linear_1d

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        with self.randomizer.fork_rng(enable_cpu=True):
            fan_in, fan_out = self.in_features, self.out_features
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            if self.bias is not None:
                bias_initializer(self.bias, fan_in=fan_in)

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        assert input_.shape[-1] == self.weight.shape[-1], \
            'Invalid shapes in Linear1D_Col forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input_.shape, self.weight.shape, self.weight.shape[-1])

        # Set up backprop all-reduce.
        input_parallel = input_

        # Matrix multiply.
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = linear_with_async_comm(input_parallel, self.weight, bias, self.process_group, True)

        if self.gather_output:
            # All-gather across the partitions.
            output = gather_forward_split_backward(output_parallel, dim=-1, process_group=self.process_group)
        else:
            output = output_parallel

        if self.skip_bias_add:
            return output, self.bias
        else:
            return output


class Linear1D_Row(ParallelModule):
    r""" Linear layer with row parallelism

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (`torch.dtype`): The dtype of parameters, defaults to None.
        parallel_input (bool): If set to ``True``, it's assumed that the input is split, defaults to False.
        skip_bias_add (bool): If set to ``True``, it will skip bias add for linear layer,
            which is preserved for kernel fusion, defaults to False
        weight_initializer (:class:`typing.Callable`, optional):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (:class:`typing.Callable`, optional):
            The initializer of bias, defaults to xavier uniform initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 process_group: ProcessGroup = None,
                 parallel_input: bool = True,
                 skip_bias_add: bool = False,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
                 stream_chunk_num: int = 1):
        super().__init__()

        self.stream_chunk_num = stream_chunk_num

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.parallel_input = parallel_input
        self.skip_bias_add = skip_bias_add
        self.process_group = process_group
        self.num_partitions = dist.get_world_size(self.process_group)

        if skip_bias_add and not bias:
            raise ValueError('cannot skip bias addition if bias is None')

        # Parameters.
        # Initialize weight.
        factory_kwargs = {'device': device, 'dtype': dtype}
        weight = torch.empty(self.out_features, self.in_features, **factory_kwargs)
        sharded_weight = shard_colwise(weight, self.process_group)
        self.weight = sharded_tensor_to_param(sharded_weight)

        if self.stream_chunk_num > 1:
            # TODO() work for inference only
            self.chunk_weight()
        if bias:
            self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.bias = None

        # offset the seed with randomizer index and rank
        seed = torch.random.initial_seed()
        self.randomizer = create_randomizer_with_offset(seed, process_group=self.process_group)

        with self.randomizer.fork_rng(enable_cpu=True):
            self.reset_parameters(weight_initializer, bias_initializer)

    @staticmethod
    def from_native_module(module: nn.Linear, process_group: Union[ProcessGroup, List[ProcessGroup]], *args,
                           **kwargs) -> ParallelModule:
        r"""
        Convert a native PyTorch linear layer to a parallelized linear layer.
        """
        # get the attributes
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        device = module.weight.device

        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, \
                f'Expected only one process group, got {len(process_group)}.'
            process_group = process_group[0]

        linear_1d = Linear1D_Row(in_features=in_features,
                                 out_features=out_features,
                                 bias=bias,
                                 device=device,
                                 process_group=process_group,
                                 *args,
                                 **kwargs)

        # TODO: copy the sharded weights
        with torch.no_grad():
            # the weigh to the linear layer is a transpose
            # thus shard on col is equal to shard on row
            sharded_weight = shard_colwise(module.weight.data, process_group)
            linear_1d.weight.data.copy_(sharded_weight)

            if bias:
                linear_1d.bias.copy_(module.bias.data)

        return linear_1d

    def chunk_weight(self):
        self.weight_list = torch.chunk(self.weight, self.stream_chunk_num, dim=0)

    @torch.no_grad()
    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.out_features
        weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)

        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)
            if self.process_group is None:
                src_rank = 0
            else:
                src_rank = dist.distributed_c10d._get_global_rank(self.process_group, 0)

            origin_device = self.bias.device
            bias = self.bias.cuda()
            dist.broadcast(bias, src=src_rank, group=self.process_group)
            bias = bias.to(origin_device)
            self.bias.copy_(bias)

    def forward(self, input_: Tensor) -> Tensor:
        # Set up backprop all-reduce.
        if self.parallel_input:
            assert input_.shape[-1] == self.weight.shape[-1], \
                'Invalid shapes in Linear1D_Row forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input_.shape, self.weight.shape, self.weight.shape[-1])
            input_ = input_
        else:
            assert divide(input_.shape[-1], self.num_partitions) == self.weight.shape[-1], \
                'Invalid shapes in Linear1D_Row forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input_.shape, self.weight.shape, self.weight.shape[-1] * self.num_partitions)
            input_ = split_forward_gather_backward(input_, dim=-1, process_group=self.process_group)

        if self.stream_chunk_num > 1:
            if self.training:
                raise RuntimeError("use stream_chunk_num=1 in Linear1D_Row for training!")
            with torch.no_grad():
                output_parallel_list = [None for i in range(self.stream_chunk_num)]
                handle_list = []
                for i in range(self.stream_chunk_num):
                    output_parallel_list[i] = F.linear(input_, self.weight_list[i])
                    handle = torch.distributed.all_reduce(output_parallel_list[i],
                                                          group=self.process_group,
                                                          async_op=True)
                    handle_list.append(handle)
                    # output_parallel_list[i] = reduce_input(output_parallel_list[i], ParallelMode.PARALLEL_1D)
                for handle in handle_list:
                    handle.wait()
                output = torch.cat(output_parallel_list, dim=-1)
        else:
            output_parallel = F.linear(input_, self.weight)
            output = reduce_forward(output_parallel, self.process_group)

        if not self.skip_bias_add:
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            return output, self.bias
