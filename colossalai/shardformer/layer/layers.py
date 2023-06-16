#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter

from colossalai.communication import broadcast
from colossalai.context import ParallelMode, seed
from colossalai.core import global_context as gpc
from colossalai.global_variables import tensor_parallel_env as env
from colossalai.kernel import LayerNorm
from colossalai.nn import init as init
from colossalai.nn.layer.base_layer import ParallelLayer
from colossalai.nn.layer.colossalai_layer._utils import ColossalaiModule
from colossalai.nn.layer.parallel_1d._utils import get_parallel_input, reduce_grad, set_parallel_input
from colossalai.nn.layer.utils import divide, set_tensor_parallel_attribute_by_partition
from colossalai.nn.layer.vanilla import VanillaLayerNorm, VanillaPatchEmbedding
from colossalai.tensor.d_tensor.api import shard_colwise, shard_rowwise
from colossalai.utils.checkpointing import (
    broadcast_state_dict,
    gather_tensor_parallel_state_dict,
    partition_tensor_parallel_state_dict,
)
from colossalai.utils.cuda import get_current_device

from ._operation import (
    gather_forward_split_backward,
    linear_with_async_comm,
    reduce_input,
    split_forward_gather_backward,
)
from .utils import create_randomizer_with_offset

Fast_LN = None
try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNorm
    Fast_LN = FastLayerNorm
except ImportError:
    pass


class ParallelModule(nn.Module, ABC):

    @abstractmethod
    def from_native_module(module: nn.Module,
                           process_group: Union[ProcessGroup, List[ProcessGroup]] = None) -> "ParallelModule":
        """
        Convert a native PyTorch module to a parallelized module.

        Args:
            module (nn.Module): the module to be converted.
            process_group (ProcessGroup or list[ProcessGroup]): the process group(s) to be used for communication.
                If this is a list, the process group at the ith index of the list will correspond to the process group
                in the ith axis of the device mesh. Defaults to None, which means the global process group.
        """
        pass


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
        self.num_partitions = dist.get_world_size(self.process_group)

        if skip_bias_add and not bias:
            raise ValueError('cannot skip bias addition if bias is None')

        self.out_features_per_partition = divide(out_features, self.num_partitions)

        # Parameters.
        # Initialize weight.
        if device is None:
            device = get_current_device()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty(self.out_features_per_partition, self.in_features, **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(self.out_features_per_partition, **factory_kwargs))
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

        linear_1d = Linear1D_Col(in_features=in_features,
                                 out_features=out_features,
                                 bias=bias,
                                 device=device,
                                 process_group=process_group,
                                 *args,
                                 **kwargs)

        # TODO: copy the sharded weights
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
        fan_in, fan_out = self.in_features, self.out_features
        weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        assert input_.shape[-1] == self.weight.shape[-1], \
            'Invalid shapes in Linear1D_Col forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input_.shape, self.weight.shape, self.weight.shape[-1])
        # Set up backprop all-reduce.
        # input_parallel = reduce_grad(input_, ParallelMode.PARALLEL_1D)
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

        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = divide(in_features, self.num_partitions)

        # Parameters.
        # Initialize weight.
        if device is None:
            device = get_current_device()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty(self.out_features, self.input_size_per_partition, **factory_kwargs))

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
            self.bias = self.bias.cuda()
            dist.broadcast(self.bias, src=src_rank, group=self.process_group)
            self.bias = self.bias.to(origin_device)

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
            # output_parallel = linear_with_async_comm(input_, self.weight, None, ParallelMode.PARALLEL_1D, False)
            output = reduce_input(output_parallel, self.process_group)

        if not self.skip_bias_add:
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            return output, self.bias


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


class Embedding1D(ParallelModule):
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
                 device: torch.device = None,
                 process_group: ProcessGroup = None,
                 weight_initializer: Callable = init.normal_(),
                 *args,
                 **kwargs):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        self.process_group = process_group
        self.num_partitions = dist.get_world_size(process_group)
        self.embed_dim_per_partition = divide(embedding_dim, self.num_partitions)

        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs
        self.gather_output = gather_output

        if device is None:
            device = get_current_device()

        self.weight = Parameter(torch.empty((num_embeddings, self.embed_dim_per_partition), device=device, dtype=dtype))

        # offset the seed with randomizer index and rank
        seed = torch.random.initial_seed()
        self.randomizer = create_randomizer_with_offset(seed, process_group=self.process_group)

        with self.randomizer.fork_rng(enable_cpu=True):
            self.reset_parameters(weight_initializer)

    @staticmethod
    def from_native_module(module: nn.Embedding,
                           process_group: Union[ProcessGroup, List[ProcessGroup]] = None) -> "Embedding1D":
        r"""
        Build a 1D parallelized Embedding from a native nn.Embedding module.
        """
        # get the attributes
        num_embedding = module.num_embeddings
        embedding_dim = module.embedding_dim
        padding_idx = module.padding_idx
        max_norm = module.max_norm
        norm_type = module.norm_type
        scale_grad_by_freq = module.scale_grad_by_freq
        sparse = module.sparse
        dtype = module.weight.dtype
        device = module.weight.device

        # sparse is not support yet
        if sparse:
            raise NotImplementedError("The Embedding1D module does not support sparse embedding yet.")

        embedding = Embedding1D(num_embeddings=num_embedding,
                                embedding_dim=embedding_dim,
                                padding_idx=padding_idx,
                                process_group=process_group,
                                dtype=dtype,
                                device=device,
                                max_norm=max_norm,
                                norm_type=norm_type,
                                scale_grad_by_freq=scale_grad_by_freq,
                                sparse=sparse)

        # copy the weight
        with torch.no_grad():
            sharded_weight = shard_colwise(module.weight.data, process_group)
            embedding.weight.copy_(sharded_weight)

        return embedding

    def reset_parameters(self, weight_initializer) -> None:
        fan_in, fan_out = self.num_embeddings, self.embed_dim
        weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input_: Tensor) -> Tensor:
        output_parallel = F.embedding(input_, self.weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)
        output = gather_forward_split_backward(output_parallel, dim=-1, process_group=self.process_group)

        return output


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
                 device: torch.device = None,
                 process_group: ProcessGroup = None,
                 weight_initializer: Callable = init.normal_(),
                 *args,
                 **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs
        self.process_group = process_group

        tensor_parallel_size = dist.get_world_size(group=process_group)
        tensor_parallel_rank = dist.get_rank(group=process_group)

        self.num_embeddings_per_partition = divide(num_embeddings, tensor_parallel_size)
        self.num_embeddings = self.num_embeddings_per_partition
        self.vocab_start_index = tensor_parallel_rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition

        self.weight = Parameter(
            torch.empty((self.num_embeddings_per_partition, self.embed_dim), device=device, dtype=dtype))

        # offset the seed with randomizer index and rank
        seed = torch.random.initial_seed()
        self.randomizer = create_randomizer_with_offset(seed, process_group=self.process_group)

        with self.randomizer.fork_rng(enable_cpu=True):
            self.reset_parameters(weight_initializer)
        # self.reset_parameters(weight_initializer)
        # self._set_tensor_parallel_attributes()
        # set_parallel_input(False)
        # env.vocab_parallel = True

    @staticmethod
    def from_native_module(module: nn.Embedding, process_group: Union[ProcessGroup, List[ProcessGroup]], *args,
                           **kwargs) -> ParallelModule:
        r"""
        Convert a native pytorch embedding module to a parallel module.
        """
        # get the origin attributes
        num_embeddings = module.num_embeddings
        embedding_dim = module.embedding_dim
        padding_idx = module.padding_idx
        device = module.weight.device

        # ensure only one process group is used
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, \
                f'Expected only one process group, got {len(process_group)}.'
            process_group = process_group[0]

        # create the parallel module
        vocab_embedding_1d = VocabParallelEmbedding1D(num_embeddings=num_embeddings,
                                                      embedding_dim=embedding_dim,
                                                      padding_idx=padding_idx,
                                                      device=device,
                                                      process_group=process_group,
                                                      *args,
                                                      **kwargs)
        with torch.no_grad():
            # shard and slice the weight along the vocabulary(num_embeddings) dimension
            # the shape of the weight is (num_embeddings, embedding_dim)
            shard_weight = shard_rowwise(module.weight.data, process_group)
            vocab_embedding_1d.weight.data.copy_(shard_weight)

        return vocab_embedding_1d

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
        output = reduce_input(output_parallel, self.process_group)
        return output
