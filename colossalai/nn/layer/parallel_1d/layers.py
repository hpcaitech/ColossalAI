#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from collections import OrderedDict
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from colossalai.communication import broadcast
from colossalai.context import ParallelMode, seed
from colossalai.core import global_context as gpc
from colossalai.global_variables import tensor_parallel_env as env
from colossalai.kernel import LayerNorm
from colossalai.nn import init as init
from colossalai.registry import LAYERS
from colossalai.utils.checkpointing import (
    broadcast_state_dict,
    gather_tensor_parallel_state_dict,
    partition_tensor_parallel_state_dict,
)
from colossalai.utils.cuda import get_current_device

from ..base_layer import ParallelLayer
from ..colossalai_layer._utils import ColossalaiModule
from ..utils import divide, set_tensor_parallel_attribute_by_partition
from ..vanilla import VanillaLayerNorm, VanillaPatchEmbedding
from ._operation import linear_with_async_comm
from ._utils import (
    gather_forward_split_backward,
    get_parallel_input,
    reduce_grad,
    reduce_input,
    set_parallel_input,
    split_forward_gather_backward,
)

Fast_LN = None
try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNorm
    Fast_LN = FastLayerNorm
except ImportError:
    pass


@LAYERS.register_module
class Linear1D(ColossalaiModule):
    r"""Linear layer for 1D parallelism.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        gather_output (bool, optional): Whether to call all-gather on output, defaults to False.
        skip_bias_add (bool, optional): If set to ``True``, it will skip bias add for linear layer,
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
                 gather_output: bool = False,
                 skip_bias_add: bool = False,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)):
        parallel_input = get_parallel_input()
        if not parallel_input and not gather_output:
            layer = Linear1D_Col(in_features,
                                 out_features,
                                 bias=bias,
                                 dtype=dtype,
                                 skip_bias_add=skip_bias_add,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer)
        else:
            layer = Linear1D_Row(in_features,
                                 out_features,
                                 bias=bias,
                                 dtype=dtype,
                                 parallel_input=parallel_input,
                                 skip_bias_add=skip_bias_add,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer)
        super().__init__(layer)


@LAYERS.register_module
class LayerNorm1D(ColossalaiModule):
    r"""
    Layer Normalization for colossalai

    Args:
        normalized_shape (int): input shape from an expected input of size.
            :math:`[* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1]
            \times \ldots \times \text{normalized_shape}[-1]]`
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps (float): a value added to the denominator for numerical stability, defaults to 1e-05.
        bias (bool, optional): Whether to add a bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
    """

    _fast_ln_supported_sizes = [
        1024, 1536, 2048, 2304, 3072, 3840, 4096, 5120, 6144, 8192, 10240, 12288, 12800, 15360, 16384, 18432, 20480,
        24576, 25600, 30720, 32768, 40960, 49152, 65536
    ]

    def __init__(self, normalized_shape: int, eps=1e-05, bias=True, dtype=None):
        if Fast_LN is not None and normalized_shape in self._fast_ln_supported_sizes:
            norm = Fast_LN(normalized_shape, eps=eps).to(dtype)
        else:
            norm = None
            try:
                from apex.normalization import FusedLayerNorm
                norm = FusedLayerNorm(normalized_shape, eps=eps).to(dtype)
            except ImportError:
                norm = LayerNorm(normalized_shape, eps=eps).to(dtype)
        super().__init__(norm)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight
            # bias
            bias = state_dict.pop(bias_key, None)
            if bias is not None:
                local_state[bias_key] = bias

        local_state = broadcast_state_dict(local_state, ParallelMode.PARALLEL_1D)
        super()._load_from_state_dict(local_state, prefix, *args)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            super()._save_to_state_dict(destination, prefix, keep_vars)


@LAYERS.register_module
class Classifier1D(ParallelLayer):
    r"""RowLinear with given weight. Classifier of 1D parallelism.

    Args:
        in_features (int): size of each input sample.
        num_classes (int): number of classes.
        weight (:class:`torch.nn.Parameter`, optional): weight of the classifier, defaults to None.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        weight_initializer (:class:`typing.Callable`, optional):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (:class:`typing.Callable`, optional):
            The initializer of bias, defaults to xavier uniform initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 weight: Parameter = None,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.parallel_input = get_parallel_input()

        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = divide(in_features, gpc.tensor_parallel_size)

        # Parameters.
        # Initialize weight.
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}
        if weight is not None:
            self.weight = weight
            self.has_weight = False
        else:
            self.weight = Parameter(torch.empty(self.num_classes, self.input_size_per_partition, **factory_kwargs))
            self.has_weight = True
        if bias:
            self.bias = Parameter(torch.empty(self.num_classes, **factory_kwargs))
        else:
            self.bias = None
        with seed(ParallelMode.TENSOR):
            self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()
        set_parallel_input(False)
        env.vocab_parallel = False

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.num_classes
        if self.has_weight:
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)
            broadcast(self.bias, gpc.get_ranks_in_group(ParallelMode.PARALLEL_1D)[0], ParallelMode.PARALLEL_1D)

    def _set_tensor_parallel_attributes(self):
        if self.has_weight:
            num_partition = gpc.get_world_size(ParallelMode.TENSOR)
            set_tensor_parallel_attribute_by_partition(self.weight, num_partition)

    def _load_from_global_state_dict(self, state_dict, prefix, *args):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            if self.has_weight:
                weight = state_dict.pop(weight_key, None)
                if weight is not None:
                    local_state[weight_key] = weight
            # bias
            if self.bias is not None:
                bias = state_dict.pop(bias_key, None)
                if bias is not None:
                    local_state[bias_key] = bias

        local_state = partition_tensor_parallel_state_dict(local_state,
                                                           ParallelMode.PARALLEL_1D,
                                                           dims={
                                                               weight_key: -1,
                                                               bias_key: 0
                                                           },
                                                           partition_states={
                                                               weight_key: True,
                                                               bias_key: False
                                                           })
        super()._load_from_global_state_dict(local_state, prefix, *args)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        local_state = OrderedDict()
        if self.has_weight:
            local_state[weight_key] = self.weight
        if self.bias is not None:
            local_state[bias_key] = self.bias
        local_state = gather_tensor_parallel_state_dict(local_state,
                                                        ParallelMode.PARALLEL_1D,
                                                        dims={
                                                            weight_key: -1,
                                                            bias_key: 0
                                                        },
                                                        partition_states={
                                                            weight_key: True,
                                                            bias_key: False
                                                        },
                                                        keep_vars=keep_vars)
        destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        # Set up backprop all-reduce.
        if self.parallel_input:
            assert input_.shape[-1] == self.weight.shape[-1], \
                'Invalid shapes in Classifier1D forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input_.shape, self.weight.shape, self.weight.shape[-1])
            input_ = input_
        else:
            assert divide(input_.shape[-1], gpc.tensor_parallel_size) == self.weight.shape[-1], \
                'Invalid shapes in Classifier1D forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input_.shape, self.weight.shape, self.weight.shape[-1] * gpc.tensor_parallel_size)
            input_ = split_forward_gather_backward(input_, ParallelMode.PARALLEL_1D, dim=-1)

        output_parallel = F.linear(input_, self.weight)
        output = reduce_input(output_parallel, ParallelMode.PARALLEL_1D)
        if self.bias is not None:
            output = output + self.bias
        return output


@LAYERS.register_module
class VocabParallelClassifier1D(ParallelLayer):
    r"""ColLinear with given weight. Classifier of 1D parallelism.

    Args:
        in_features (int): size of each input sample.
        num_classes (int): number of classes.
        weight (:class:`torch.nn.Parameter`, optional): weight of the classifier, defaults to None.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        weight_initializer (:class:`typing.Callable`, optional):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (:class:`typing.Callable`, optional):
            The initializer of bias, defaults to xavier uniform initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 weight: Parameter = None,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 gather_output: bool = False,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.gather_output = gather_output
        self.parallel_input = get_parallel_input()

        # Divide the weight matrix along the last dimension.
        self.num_classes_per_partition = divide(num_classes, gpc.tensor_parallel_size)

        # Parameters.
        # Initialize weight.
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}
        if weight is not None:
            self.weight = weight
            self.has_weight = False
        else:
            self.weight = Parameter(torch.empty(self.num_classes_per_partition, self.in_features, **factory_kwargs))
            self.has_weight = True
        if bias:
            self.bias = Parameter(torch.empty(self.num_classes_per_partition, **factory_kwargs))
        else:
            self.bias = None
        with seed(ParallelMode.TENSOR):
            self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()
        set_parallel_input(False)
        env.vocab_parallel = True

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.num_classes
        if self.has_weight:
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)

    def _set_tensor_parallel_attributes(self):
        num_partition = gpc.get_world_size(ParallelMode.TENSOR)
        if self.has_weight:
            set_tensor_parallel_attribute_by_partition(self.weight, num_partition)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, num_partition)

    def _load_from_global_state_dict(self, state_dict, prefix, *args):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            if self.has_weight:
                weight = state_dict.pop(weight_key, None)
                if weight is not None:
                    local_state[weight_key] = weight
            # bias
            if self.bias is not None:
                bias = state_dict.pop(bias_key, None)
                if bias is not None:
                    local_state[bias_key] = bias

        local_state = partition_tensor_parallel_state_dict(local_state,
                                                           ParallelMode.PARALLEL_1D,
                                                           dims={
                                                               weight_key: 0,
                                                               bias_key: 0
                                                           },
                                                           partition_states={
                                                               weight_key: True,
                                                               bias_key: True
                                                           })
        super()._load_from_global_state_dict(local_state, prefix, *args)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        local_state = OrderedDict()
        if self.has_weight:
            local_state[weight_key] = self.weight
        if self.bias is not None:
            local_state[bias_key] = self.bias
        local_state = gather_tensor_parallel_state_dict(local_state,
                                                        ParallelMode.PARALLEL_1D,
                                                        dims={
                                                            weight_key: 0,
                                                            bias_key: 0
                                                        },
                                                        partition_states={
                                                            weight_key: True,
                                                            bias_key: True
                                                        },
                                                        keep_vars=keep_vars)
        destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        assert input_.shape[-1] == self.weight.shape[-1], \
            'Invalid shapes in VocabParallelClassifier1D forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input_.shape, self.weight.shape, self.weight.shape[-1])
        # Set up backprop all-reduce.
        input_parallel = reduce_grad(input_, ParallelMode.PARALLEL_1D)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_forward_split_backward(output_parallel, ParallelMode.PARALLEL_1D, dim=-1)
        else:
            output = output_parallel
        return output


@LAYERS.register_module
class Linear1D_Col(ParallelLayer):
    r"""Linear layer with column parallelism.

    The linear layer is defined as :math:`Y = XA + b`. A is parallelized along
    its second dimension as :math:`A = [A_1, ..., A_p]`.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        gather_output (bool, optional): If true, call all-gather on output and make Y available
                    to all GPUs, otherwise, every GPU will have its output
                    which is :math:`Y_i = XA_i`, defaults to False
        skip_bias_add (bool, optional): If set to ``True``, it will skip bias add for linear layer,
            which is preserved for kernel fusion, defaults to Fals
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

        if skip_bias_add and not bias:
            raise ValueError('cannot skip bias addition if bias is None')

        self.out_features_per_partition = divide(out_features, gpc.tensor_parallel_size)

        # Parameters.
        # Initialize weight.
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}
        self.weight = Parameter(torch.empty(self.out_features_per_partition, self.in_features, **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(self.out_features_per_partition, **factory_kwargs))
        else:
            self.bias = None
        with seed(ParallelMode.TENSOR):
            self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()
        is_parallel_output = not self.gather_output
        set_parallel_input(is_parallel_output)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.out_features
        weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)

    def _set_tensor_parallel_attributes(self):
        num_partition = gpc.get_world_size(ParallelMode.TENSOR)
        set_tensor_parallel_attribute_by_partition(self.weight, num_partition)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, num_partition)

    def _load_from_global_state_dict(self, state_dict, prefix, *args):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight
            # bias
            if self.bias is not None:
                bias = state_dict.pop(bias_key, None)
                if bias is not None:
                    local_state[bias_key] = bias

        local_state = partition_tensor_parallel_state_dict(local_state,
                                                           ParallelMode.PARALLEL_1D,
                                                           dims={
                                                               weight_key: 0,
                                                               bias_key: 0
                                                           },
                                                           partition_states={
                                                               weight_key: True,
                                                               bias_key: True
                                                           })
        super()._load_from_global_state_dict(local_state, prefix, *args)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        local_state = OrderedDict({weight_key: self.weight})
        if self.bias is not None:
            local_state[bias_key] = self.bias
        local_state = gather_tensor_parallel_state_dict(local_state,
                                                        ParallelMode.PARALLEL_1D,
                                                        dims={
                                                            weight_key: 0,
                                                            bias_key: 0
                                                        },
                                                        partition_states={
                                                            weight_key: True,
                                                            bias_key: True
                                                        },
                                                        keep_vars=keep_vars)
        destination.update(local_state)

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        assert input_.shape[-1] == self.weight.shape[-1], \
            'Invalid shapes in Linear1D_Col forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input_.shape, self.weight.shape, self.weight.shape[-1])
        # Set up backprop all-reduce.
        # input_parallel = reduce_grad(input_, ParallelMode.PARALLEL_1D)
        input_parallel = input_
        # Matrix multiply.
        bias = self.bias if not self.skip_bias_add else None
        # output_parallel = F.linear(input_parallel, self.weight, bias)
        output_parallel = linear_with_async_comm(input_parallel, self.weight, bias, ParallelMode.PARALLEL_1D, True)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_forward_split_backward(output_parallel, ParallelMode.PARALLEL_1D, dim=-1)
        else:
            output = output_parallel

        if self.skip_bias_add:
            return output, self.bias
        else:
            return output


@LAYERS.register_module
class Linear1D_Row(ParallelLayer):
    r""" Linear layer with row parallelism

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        parallel_input (bool, optional): If set to ``True``, it's assumed that the input is split, defaults to False.
        skip_bias_add (bool, optional): If set to ``True``, it will skip bias add for linear layer,
            which is preserved for kernel fusion, defaults to Fals
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

        if skip_bias_add and not bias:
            raise ValueError('cannot skip bias addition if bias is None')

        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = divide(in_features, gpc.tensor_parallel_size)

        # Parameters.
        # Initialize weight.
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}
        self.weight = Parameter(torch.empty(self.out_features, self.input_size_per_partition, **factory_kwargs))

        if self.stream_chunk_num > 1:
            # TODO() work for inference only
            self.chunk_weight()
        if bias:
            self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.bias = None
        with seed(ParallelMode.TENSOR):
            self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()
        set_parallel_input(False)

    def chunk_weight(self):
        self.weight_list = torch.chunk(self.weight, self.stream_chunk_num, dim=0)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.out_features
        weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)
            broadcast(self.bias, gpc.get_ranks_in_group(ParallelMode.PARALLEL_1D)[0], ParallelMode.PARALLEL_1D)

    def _set_tensor_parallel_attributes(self):
        num_partition = gpc.get_world_size(ParallelMode.TENSOR)
        set_tensor_parallel_attribute_by_partition(self.weight, num_partition)

    def _load_from_global_state_dict(self, state_dict, prefix, *args):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight
            # bias
            if self.bias is not None:
                bias = state_dict.pop(bias_key, None)
                if bias is not None:
                    local_state[bias_key] = bias

        local_state = partition_tensor_parallel_state_dict(local_state,
                                                           ParallelMode.PARALLEL_1D,
                                                           dims={
                                                               weight_key: -1,
                                                               bias_key: 0
                                                           },
                                                           partition_states={
                                                               weight_key: True,
                                                               bias_key: False
                                                           })
        super()._load_from_global_state_dict(local_state, prefix, *args)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        local_state = OrderedDict({weight_key: self.weight})
        if self.bias is not None:
            local_state[bias_key] = self.bias
        local_state = gather_tensor_parallel_state_dict(local_state,
                                                        ParallelMode.PARALLEL_1D,
                                                        dims={
                                                            weight_key: -1,
                                                            bias_key: 0
                                                        },
                                                        partition_states={
                                                            weight_key: True,
                                                            bias_key: False
                                                        },
                                                        keep_vars=keep_vars)
        destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        # Set up backprop all-reduce.
        if self.parallel_input:
            assert input_.shape[-1] == self.weight.shape[-1], \
                'Invalid shapes in Linear1D_Row forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input_.shape, self.weight.shape, self.weight.shape[-1])
            input_ = input_
        else:
            assert divide(input_.shape[-1], gpc.tensor_parallel_size) == self.weight.shape[-1], \
                'Invalid shapes in Linear1D_Row forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input_.shape, self.weight.shape, self.weight.shape[-1] * gpc.tensor_parallel_size)
            input_ = split_forward_gather_backward(input_, ParallelMode.PARALLEL_1D, dim=-1)

        if self.stream_chunk_num > 1:
            if self.training:
                raise RuntimeError("use stream_chunk_num=1 in Linear1D_Row for training!")
            with torch.no_grad():
                output_parallel_list = [None for i in range(self.stream_chunk_num)]
                handle_list = []
                for i in range(self.stream_chunk_num):
                    output_parallel_list[i] = F.linear(input_, self.weight_list[i])
                    handle = torch.distributed.all_reduce(output_parallel_list[i],
                                                          group=gpc.get_group(ParallelMode.PARALLEL_1D),
                                                          async_op=True)
                    handle_list.append(handle)
                    # output_parallel_list[i] = reduce_input(output_parallel_list[i], ParallelMode.PARALLEL_1D)
                for handle in handle_list:
                    handle.wait()
                output = torch.cat(output_parallel_list, dim=-1)
        else:
            output_parallel = F.linear(input_, self.weight)
            # output_parallel = linear_with_async_comm(input_, self.weight, None, ParallelMode.PARALLEL_1D, False)
            output = reduce_input(output_parallel, ParallelMode.PARALLEL_1D)
        if not self.skip_bias_add:
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            return output, self.bias


@LAYERS.register_module
class Embedding1D(ParallelLayer):
    r"""Embedding for 1D parallelism.

    Args:
        num_embeddings (int): number of embeddings.
        embedding_dim (int): dimension of embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx do not contribute to the gradient;
            therefore, the embedding vector at padding_idx is not updated during training,
            i.e. it remains as a fixed “pad”, defaults to None.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        weight_initializer (:class:`typing.Callable`, optional):
            he initializer of weight, defaults to normal initializer.

    The ``args`` and ``kwargs`` used in :class:`torch.nn.functional.embedding` should contain:
    ::

        max_norm (float, optional): If given, each embedding vector with norm larger than max_norm is
                    renormalized to have norm max_norm. Note: this will modify weight in-place.
        norm_type (float, optional): The p of the p-norm to compute for the max_norm option. Default 2.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse
                    of frequency of the words in the mini-batch. Default False.
        sparse (bool, optional): If True, gradient w.r.t. weight will be a sparse tensor. Default False.

    More details about ``args`` and ``kwargs`` could be found in
    `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html#torch.nn.functional.embedding>`_.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int = None,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.normal_(),
                 *args,
                 **kwargs):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        embed_dim_per_partition = divide(embedding_dim, gpc.tensor_parallel_size)

        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        self.weight = Parameter(
            torch.empty((num_embeddings, embed_dim_per_partition), device=get_current_device(), dtype=dtype))

        self.reset_parameters(weight_initializer)
        self._set_tensor_parallel_attributes()
        set_parallel_input(False)

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, gpc.tensor_parallel_size)

    def reset_parameters(self, weight_initializer) -> None:
        with seed(ParallelMode.TENSOR):
            fan_in, fan_out = self.num_embeddings, self.embed_dim
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def _load_from_global_state_dict(self, state_dict, prefix, *args):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight

        local_state = partition_tensor_parallel_state_dict(local_state,
                                                           ParallelMode.PARALLEL_1D,
                                                           dims={weight_key: -1},
                                                           partition_states={weight_key: True})
        super()._load_from_global_state_dict(local_state, prefix, *args)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        local_state = OrderedDict({weight_key: self.weight})
        local_state = gather_tensor_parallel_state_dict(local_state,
                                                        ParallelMode.PARALLEL_1D,
                                                        dims={weight_key: -1},
                                                        partition_states={weight_key: True},
                                                        keep_vars=keep_vars)
        destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:

        output_parallel = F.embedding(input_, self.weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)

        output = gather_forward_split_backward(output_parallel, ParallelMode.PARALLEL_1D, dim=-1)

        return output


@LAYERS.register_module
class VocabParallelEmbedding1D(ParallelLayer):
    r"""Embedding parallelized in the vocabulary dimension.

    Args:
        num_embeddings (int): number of embeddings.
        embedding_dim (int): dimension of embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx do not contribute to the gradient;
            therefore, the embedding vector at padding_idx is not updated during training,
            i.e. it remains as a fixed “pad”, defaults to None.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        weight_initializer (:class:`typing.Callable`, optional):
            he initializer of weight, defaults to normal initializer.

    The ``args`` and ``kwargs`` used in :class:``torch.nn.functional.embedding`` should contain:
    ::

        max_norm (float, optional): If given, each embedding vector with norm larger than max_norm is
                    renormalized to have norm max_norm. Note: this will modify weight in-place.
        norm_type (float, optional): The p of the p-norm to compute for the max_norm option. Default 2.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse
                    of frequency of the words in the mini-batch. Default False.
        sparse (bool, optional): If True, gradient w.r.t. weight will be a sparse tensor. Default False.

    More details about ``args`` and ``kwargs`` could be found in
    `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html#torch.nn.functional.embedding>`_.

    More details about initializer please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int = None,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.normal_(),
                 *args,
                 **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        tensor_parallel_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
        tensor_parallel_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
        self.num_embeddings_per_partition = divide(num_embeddings, tensor_parallel_size)
        self.vocab_start_index = tensor_parallel_rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition

        self.weight = Parameter(
            torch.empty((self.num_embeddings_per_partition, self.embed_dim), device=get_current_device(), dtype=dtype))

        self.reset_parameters(weight_initializer)
        self._set_tensor_parallel_attributes()
        set_parallel_input(False)
        env.vocab_parallel = True

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, gpc.tensor_parallel_size)

    def reset_parameters(self, weight_initializer) -> None:
        with seed(ParallelMode.TENSOR):
            fan_in, fan_out = self.num_embeddings, self.embed_dim
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None and \
                self.padding_idx >= self.vocab_start_index and self.padding_idx < self.vocab_end_index:
            with torch.no_grad():
                self.weight[self.padding_idx - self.vocab_start_index].fill_(0)

    def _load_from_global_state_dict(self, state_dict, prefix, *args):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight

        local_state = partition_tensor_parallel_state_dict(local_state,
                                                           ParallelMode.PARALLEL_1D,
                                                           dims={weight_key: 0},
                                                           partition_states={weight_key: True})
        super()._load_from_global_state_dict(local_state, prefix, *args)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        local_state = OrderedDict({weight_key: self.weight})
        local_state = gather_tensor_parallel_state_dict(local_state,
                                                        ParallelMode.PARALLEL_1D,
                                                        dims={weight_key: 0},
                                                        partition_states={weight_key: True},
                                                        keep_vars=keep_vars)
        destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0

        output_parallel = F.embedding(masked_input, self.weight, self.padding_idx, *self.embed_args,
                                      **self.embed_kwargs)

        # Mask the output embedding.
        output_parallel[input_mask, :] = 0.
        # Reduce across all the model parallel GPUs.
        output = reduce_input(output_parallel, ParallelMode.PARALLEL_1D)
        return output


@LAYERS.register_module
class Dropout1D(ParallelLayer):
    """Dropout layer of 1D parallelism.

    Args:
        p (float, optional): probability of an element to be zeroed, defaults 0.5.
        inplace (bool, optional): whether to do dropout in-place, default to be False.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.parallel_input = get_parallel_input()
        self.p = p
        self.inplace = inplace

    def forward(self, input_: Tensor) -> Tensor:
        if self.parallel_input:
            with seed(ParallelMode.TENSOR):
                output = F.dropout(input_, self.p, self.training, self.inplace)
        else:
            output = F.dropout(input_, self.p, self.training, self.inplace)
        return output


@LAYERS.register_module
class PatchEmbedding1D(ColossalaiModule):
    """
    2D Image to Patch Embedding

    :param img_size: image size
    :type img_size: int
    :param patch_size: patch size
    :type patch_size: int
    :param in_chans: number of channels of input image
    :type in_chans: int
    :param embed_size: size of embedding
    :type embed_size: int
    :param dtype: The dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    :param flatten: whether to flatten output tensor, defaults to True
    :type flatten: bool, optional
    :param weight_initializer: The intializer of weight, defaults to kaiming uniform initializer
    :type weight_initializer: typing.Callable, optional
    :param bias_initializer: The intializer of bias, defaults to xavier uniform initializer
    :type bias_initializer: typing.Callable, optional
    :param position_embed_initializer: The intializer of position embedding, defaults to zero
    :type position_embed_initializer: typing.Callable, optional
    """

    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 embed_size: int,
                 dtype: torch.dtype = None,
                 flatten: bool = True,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
                 position_embed_initializer: Callable = init.zeros_()):
        embed = VanillaPatchEmbedding(img_size,
                                      patch_size,
                                      in_chans,
                                      embed_size,
                                      dtype=dtype,
                                      flatten=flatten,
                                      weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer,
                                      position_embed_initializer=position_embed_initializer)
        super().__init__(embed)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        local_state = OrderedDict()
        param_keys = [prefix + 'weight', prefix + 'bias', prefix + 'cls_token', prefix + 'pos_embed']
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            for key in param_keys:
                param = state_dict.pop(key, None)
                if param is not None:
                    local_state[key] = param

        local_state = broadcast_state_dict(local_state, ParallelMode.PARALLEL_1D)
        super()._load_from_state_dict(local_state, prefix, *args)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            super()._save_to_state_dict(destination, prefix, keep_vars)
