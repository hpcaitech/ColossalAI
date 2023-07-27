import torch
import torch.nn as nn
import torch.nn.functional as F
from coati.models.lora import LoraLinear
from torch import Tensor

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.layer.parallel_1d._utils import (
    gather_forward_split_backward,
    reduce_grad,
    reduce_input,
    split_forward_gather_backward,
)
from colossalai.nn.layer.utils import divide


def linear_1d_col_fn(self: nn.Linear, input_: Tensor, gather_output: bool = False) -> Tensor:
    assert input_.shape[-1] == self.weight.shape[-1], \
        'Invalid shapes in Linear1D_Col forward: input={}, weight={}. Expected last dim of input {}.'.format(
        input_.shape, self.weight.shape, self.weight.shape[-1])
    # Set up backprop all-reduce.
    # TODO(ver217): this relies on GPC
    input_parallel = reduce_grad(input_, ParallelMode.PARALLEL_1D)
    # Matrix multiply.

    output_parallel = F.linear(input_parallel, self.weight, self.bias)
    if gather_output:
        # All-gather across the partitions.
        output = gather_forward_split_backward(output_parallel, ParallelMode.PARALLEL_1D, dim=-1)
    else:
        output = output_parallel
    return output


def linear_1d_row_fn(self: nn.Linear, input_: Tensor, parallel_input: bool = True) -> Tensor:
    # Set up backprop all-reduce.
    if parallel_input:
        assert input_.shape[-1] == self.weight.shape[-1], \
            'Invalid shapes in Linear1D_Row forward: input={}, weight={}. Expected last dim of input {}.'.format(
            input_.shape, self.weight.shape, self.weight.shape[-1])
        input_ = input_
    else:
        assert divide(input_.shape[-1], gpc.tensor_parallel_size) == self.weight.shape[-1], \
            'Invalid shapes in Linear1D_Row forward: input={}, weight={}. Expected last dim of input {}.'.format(
            input_.shape, self.weight.shape, self.weight.shape[-1] * gpc.tensor_parallel_size)
        input_ = split_forward_gather_backward(input_, ParallelMode.PARALLEL_1D, dim=-1)

    output_parallel = F.linear(input_, self.weight)
    output = reduce_input(output_parallel, ParallelMode.PARALLEL_1D)
    if self.bias is not None:
        output = output + self.bias
    return output


def lora_linear_1d_col_fn(self: LoraLinear, input_: Tensor, gather_output: bool = False) -> Tensor:
    assert input_.shape[-1] == self.weight.shape[-1], \
        'Invalid shapes in Linear1D_Col forward: input={}, weight={}. Expected last dim of input {}.'.format(
        input_.shape, self.weight.shape, self.weight.shape[-1])
    # Set up backprop all-reduce.
    # TODO(ver217): this relies on GPC
    input_parallel = reduce_grad(input_, ParallelMode.PARALLEL_1D)
    # Matrix multiply.

    output_parallel = F.linear(input_parallel, self.weight, self.bias)

    if gather_output:
        # All-gather across the partitions.
        output = gather_forward_split_backward(output_parallel, ParallelMode.PARALLEL_1D, dim=-1)
    else:
        output = output_parallel

    if self.r > 0:
        lora_addon = (self.lora_dropout(input_) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        if not gather_output:
            lora_addon = split_forward_gather_backward(lora_addon, ParallelMode.PARALLEL_1D, dim=-1)
        output = output + lora_addon
    return output


def lora_linear_1d_row_fn(self: LoraLinear, input_: Tensor, parallel_input: bool = True) -> Tensor:
    # Set up backprop all-reduce.
    if parallel_input:
        assert input_.shape[-1] == self.weight.shape[-1], \
            'Invalid shapes in Linear1D_Row forward: input={}, weight={}. Expected last dim of input {}.'.format(
            input_.shape, self.weight.shape, self.weight.shape[-1])
        if self.r > 0:
            lora_input = gather_forward_split_backward(input_, ParallelMode.PARALLEL_1D, dim=-1)
        input_ = input_
    else:
        assert divide(input_.shape[-1], gpc.tensor_parallel_size) == self.weight.shape[-1], \
            'Invalid shapes in Linear1D_Row forward: input={}, weight={}. Expected last dim of input {}.'.format(
            input_.shape, self.weight.shape, self.weight.shape[-1] * gpc.tensor_parallel_size)
        if self.r > 0:
            lora_input = input_
        input_ = split_forward_gather_backward(input_, ParallelMode.PARALLEL_1D, dim=-1)

    output_parallel = F.linear(input_, self.weight)
    output = reduce_input(output_parallel, ParallelMode.PARALLEL_1D)
    if self.bias is not None:
        output = output + self.bias
    if self.r > 0:
        lora_addon = (self.lora_dropout(lora_input) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        output = output + lora_addon
    return output


def vocab_parallel_embedding_fn(self: nn.Embedding, input_: Tensor) -> Tensor:
    tp_size = gpc.tensor_parallel_size
    tp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
    vocab_size_per_partition = divide(self.num_embeddings, tp_size)
    vocab_start_index = tp_rank * vocab_size_per_partition
    vocab_end_index = vocab_start_index + vocab_size_per_partition
    # Build the mask.
    input_mask = (input_ < vocab_start_index) | (input_ >= vocab_end_index)
    # Mask the input.
    masked_input = input_.clone() - vocab_start_index
    masked_input[input_mask] = 0

    output_parallel = F.embedding(masked_input, self.weight, self.padding_idx, self.max_norm, self.norm_type,
                                  self.scale_grad_by_freq, self.sparse)

    # Mask the output embedding.
    output_parallel[input_mask, :] = 0.
    # Reduce across all the model parallel GPUs.
    output = reduce_input(output_parallel, ParallelMode.PARALLEL_1D)
    return output


def vocab_parallel_lm_head_fn(self: nn.Linear, input_: Tensor, gather_output: bool = True) -> Tensor:
    assert input_.shape[-1] == self.weight.shape[-1], \
        'Invalid shapes in VocabParallelLMHead1D forward: input={}, weight={}. Expected last dim of input {}.'.format(
            input_.shape, self.weight.shape, self.weight.shape[-1])
    # Set up backprop all-reduce.
    input_parallel = reduce_grad(input_, ParallelMode.PARALLEL_1D)
    # Matrix multiply.
    output_parallel = F.linear(input_parallel, self.weight, self.bias)
    if gather_output:
        # All-gather across the partitions.
        output = gather_forward_split_backward(output_parallel, ParallelMode.PARALLEL_1D, dim=-1)
    else:
        output = output_parallel
    return output
