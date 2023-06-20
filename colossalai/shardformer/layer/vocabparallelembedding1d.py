#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from collections import OrderedDict
from typing import Callable, List, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter

from colossalai.context import ParallelMode, seed
from colossalai.nn import init as init
from colossalai.nn.layer.base_layer import ParallelLayer
from colossalai.nn.layer.utils import divide
from colossalai.tensor.d_tensor.api import shard_rowwise
from colossalai.utils.checkpointing import gather_tensor_parallel_state_dict

from ._operation import reduce_input
from .parallelmodule import ParallelModule
from .utils import create_randomizer_with_offset

Fast_LN = None
try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNorm
    Fast_LN = FastLayerNorm
except ImportError:
    pass


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
