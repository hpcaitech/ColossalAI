#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.distributed as dist
from colossalai.core import global_context as gpc
from colossalai.global_variables import tensor_parallel_env as env

from ..utils import divide


def set_parallel_input(input_parallel: bool):
    env.parallel_input_1d = input_parallel


def get_parallel_input():
    return env.parallel_input_1d


def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank):
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f, index_l


def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank)


def _reduce(input_, parallel_mode):
    # skip if only one rank involved
    if gpc.get_world_size(parallel_mode) == 1:
        return input_
    group = gpc.get_cpu_group(parallel_mode) if input_.device.type == "cpu" else gpc.get_group(parallel_mode)
    dist.all_reduce(input_, group=group)

    return input_


def _split(input_, parallel_mode, dim=-1):
    # skip if only one rank involved
    world_size = gpc.get_world_size(parallel_mode)
    if world_size == 1:
        return input_

    # Split along last dimension.
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, \
        f'The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), ' \
        f'cannot split tensor evenly'

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    rank = gpc.get_local_rank(parallel_mode)
    output = tensor_list[rank].contiguous()

    return output


def _gather(input_, parallel_mode, dim=-1):
    # skip if only one rank involved
    world_size = gpc.get_world_size(parallel_mode)
    if world_size == 1:
        return input_

    # all gather
    rank = gpc.get_local_rank(parallel_mode)
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    group = gpc.get_cpu_group(parallel_mode) if input_.device.type == "cpu" else gpc.get_group(parallel_mode)
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


class _ReduceGrad(torch.autograd.Function):
    """
    Pass the input to the model parallel region.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
    """

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_, parallel_mode):
        ctx.mode = parallel_mode
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.mode), None


class _ReduceInput(torch.autograd.Function):
    """
    All-reduce the input from the model parallel region.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
    """

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_, parallel_mode):
        return _reduce(input_, parallel_mode)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Split the input and keep only the corresponding chuck to the rank.
    
    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_, parallel_mode, dim):
        ctx.mode = parallel_mode
        ctx.dim = dim
        return _split(input_, parallel_mode, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.mode, ctx.dim), None, None


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)

    @staticmethod
    def forward(ctx, input_, parallel_mode, dim):
        ctx.mode = parallel_mode
        ctx.dim = dim
        return _gather(input_, parallel_mode, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.mode, ctx.dim), None, None


def reduce_grad(input_, parallel_mode):
    return _ReduceGrad.apply(input_, parallel_mode)


def reduce_input(input_, parallel_mode):
    return _ReduceInput.apply(input_, parallel_mode)


def split_forward_gather_backward(input_, parallel_mode, dim):
    return _SplitForwardGatherBackward.apply(input_, parallel_mode, dim)


def gather_forward_split_backward(input_, parallel_mode, dim):
    return _GatherForwardSplitBackward.apply(input_, parallel_mode, dim)
