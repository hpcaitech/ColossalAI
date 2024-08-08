#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Callable, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from colossalai.lazy import LazyInitContext
from colossalai.nn import init as init
from colossalai.nn.layer.utils import divide
from colossalai.tensor.d_tensor.api import (
    is_distributed_tensor,
    shard_colwise,
    shard_rowwise,
    sharded_tensor_to_existing_param,
)

from ._operation import gather_forward_split_backward, reduce_forward
from .parallel_module import PaddingParallelModule, ParallelModule
from .utils import create_randomizer_with_offset

__all__ = ["Embedding1D", "VocabParallelEmbedding1D", "PaddingEmbedding"]


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

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
        process_group: ProcessGroup = None,
        gather_output: bool = True,
        weight: Optional[nn.Parameter] = None,
        weight_initializer: Callable = init.normal_(),
        fp8_communication: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.process_group = process_group

        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs
        self.gather_output = gather_output
        self.fp8_communication = fp8_communication

        # offset the seed with randomizer index and rank
        seed = torch.random.initial_seed()
        self.randomizer = create_randomizer_with_offset(seed, process_group=self.process_group)

        # Parameters.
        if weight is None:
            factory_kwargs = {"device": device, "dtype": dtype}
            self.weight = nn.Parameter(torch.empty((num_embeddings, self.embedding_dim), **factory_kwargs))
        else:
            weight.data = weight.data.to(device=device, dtype=dtype)
            self.weight = weight
        if not is_distributed_tensor(self.weight):
            sharded_weight = shard_colwise(self.weight.data, process_group)
            sharded_tensor_to_existing_param(sharded_weight, self.weight)

        if weight is None:
            with self.randomizer.fork_rng(enable_cpu=True):
                self.reset_parameters(weight_initializer)

    @staticmethod
    def from_native_module(
        module: nn.Embedding, process_group: Union[ProcessGroup, List[ProcessGroup]] = None, *args, **kwargs
    ) -> "Embedding1D":
        r"""
        Build a 1D parallelized Embedding from a native nn.Embedding module.
        """
        LazyInitContext.materialize(module)
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

        embedding = Embedding1D(
            num_embeddings=num_embedding,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            process_group=process_group,
            dtype=dtype,
            device=device,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            weight=module.weight,
            *args,
            **kwargs,
        )

        return embedding

    def reset_parameters(self, weight_initializer) -> None:
        fan_in, fan_out = self.num_embeddings, self.embedding_dim
        weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input_: Tensor) -> Tensor:
        output_parallel = F.embedding(input_, self.weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)
        if self.gather_output:
            output = gather_forward_split_backward(
                output_parallel, dim=-1, process_group=self.process_group, fp8_communication=self.fp8_communication
            )
            return output
        else:
            return output_parallel


class PaddingEmbedding(PaddingParallelModule):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
        weight: Optional[nn.Parameter] = None,
        make_vocab_size_divisible_by: int = 64,
        *args,
        **kwargs,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embed_args = args
        self.embed_kwargs = kwargs
        self.padding_idx = padding_idx
        if num_embeddings % make_vocab_size_divisible_by != 0:
            self.num_embeddings = (
                num_embeddings + make_vocab_size_divisible_by - (num_embeddings % make_vocab_size_divisible_by)
            )
        # create weight and bias
        if weight is None:
            factory_kwargs = {"device": device, "dtype": dtype}
            weight = nn.Parameter(torch.empty((num_embeddings, self.embedding_dim), **factory_kwargs))
        else:
            weight.data = weight.data.to(device=device, dtype=dtype)

        super().__init__(self.num_embeddings, num_embeddings, weight)

        if weight is None:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(input, self.weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)

    @staticmethod
    def from_native_module(
        module: nn.Embedding, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> PaddingParallelModule:
        r"""
        Convert a native pytorch embedding module to a parallel module.
        """
        LazyInitContext.materialize(module)
        # get the origin attributes
        num_embeddings = module.num_embeddings
        embedding_dim = module.embedding_dim
        padding_idx = module.padding_idx
        device = module.weight.device
        # create the parallel module
        padding_embedding = PaddingEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            device=device,
            weight=module.weight,
            *args,
            **kwargs,
        )

        return padding_embedding


class VocabParallelEmbedding1D(PaddingParallelModule):
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

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
        process_group: ProcessGroup = None,
        weight: Optional[nn.Parameter] = None,
        weight_initializer: Callable = init.normal_(),
        make_vocab_size_divisible_by: int = 64,
        fp8_communication: bool = False,
        *args,
        **kwargs,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embed_args = args
        self.embed_kwargs = kwargs
        self.process_group = process_group
        self.fp8_communication = fp8_communication

        tensor_parallel_size = dist.get_world_size(group=process_group)
        tensor_parallel_rank = dist.get_rank(group=process_group)

        # generate weight and bias
        if weight is None:
            factory_kwargs = {"device": device, "dtype": dtype}
            weight = nn.Parameter(torch.empty((num_embeddings, self.embedding_dim), **factory_kwargs))
        else:
            weight.data = weight.data.to(device=device, dtype=dtype)

        # calculate new padding size
        multiple = make_vocab_size_divisible_by * tensor_parallel_size
        if num_embeddings % multiple != 0:
            self.num_embeddings = num_embeddings + multiple - (num_embeddings % multiple)

        # resize vocabulary size
        super().__init__(self.num_embeddings, num_embeddings, weight)

        # deal with tensor parallelism
        self.num_embeddings_per_partition = divide(self.num_embeddings, tensor_parallel_size)
        self.vocab_start_index = tensor_parallel_rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition

        # padding index
        self.padding_idx = self._select_padding_idx(padding_idx)

        # offset the seed with randomizer index and rank
        seed = torch.random.initial_seed()
        self.randomizer = create_randomizer_with_offset(seed, process_group=self.process_group)

        if not is_distributed_tensor(self.weight):
            sharded_weight = shard_rowwise(self.weight.data, process_group)
            sharded_tensor_to_existing_param(sharded_weight, self.weight)

        if weight is None:
            self.reset_parameters(weight_initializer)

    @staticmethod
    def from_native_module(
        module: nn.Embedding, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> PaddingParallelModule:
        r"""
        Convert a native pytorch embedding module to a parallel module.
        """
        LazyInitContext.materialize(module)
        # get the origin attributes
        num_embeddings = module.num_embeddings
        embedding_dim = module.embedding_dim
        padding_idx = module.padding_idx
        device = module.weight.device

        # ensure only one process group is used
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, f"Expected only one process group, got {len(process_group)}."
            process_group = process_group[0]

        # create the parallel module
        vocab_embedding_1d = VocabParallelEmbedding1D(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            device=device,
            process_group=process_group,
            weight=module.weight,
            *args,
            **kwargs,
        )

        return vocab_embedding_1d

    def reset_parameters(self, weight_initializer) -> None:
        with self.randomizer.fork_rng(enable_cpu=True):
            fan_in, fan_out = self.num_embeddings, self.embedding_dim
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if (
            self.padding_idx is not None
            and self.padding_idx >= self.vocab_start_index
            and self.padding_idx < self.vocab_end_index
        ):
            with torch.no_grad():
                self.weight[self.padding_idx - self.vocab_start_index].fill_(0)

    def _select_padding_idx(self, padding_idx: int):
        # select padding index according to the rank
        if padding_idx is None:
            return None
        elif padding_idx < self.vocab_end_index and padding_idx >= self.vocab_start_index:
            return padding_idx - self.vocab_start_index
        else:
            return None

    def forward(self, input_: Tensor) -> Tensor:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        output_parallel = F.embedding(
            masked_input, self.weight, self.padding_idx, *self.embed_args, **self.embed_kwargs
        )
        # Mask the output embedding.
        embedding_output = output_parallel.clone()
        embedding_output[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_forward(embedding_output, self.process_group, fp8_communication=self.fp8_communication)
        return output
