#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch import Tensor

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device


def all_gather(tensor: Tensor, dim: int, parallel_mode: ParallelMode, async_op: bool = False) -> Tensor:
    r"""Gathers all tensors from the parallel group and concatenates them in a
    specific dimension.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be gathered.
        dim (int): The dimension concatenating in.
        parallel_mode (:class:`colossalai.context.ParallelMode`): Parallel group mode used in this communication.
        async_op (bool, optional): Whether operations are asynchronous.

    Returns:
        Union[tuple(:class:`torch.Tensor`, work handle), :class:`torch.Tensor`]: The result of all-together only,
        if async_op is set to False. A tuple of output of all-gather and Async work handle, if async_op is set to True.
    """
    depth = gpc.get_world_size(parallel_mode)
    if depth == 1:
        out = tensor
        work = None
    else:
        shape = list(tensor.shape)
        shape[0], shape[dim] = shape[dim], shape[0]
        shape[0] *= depth
        out = torch.empty(shape, dtype=tensor.dtype, device=get_current_device())
        temp = list(torch.chunk(out, depth, dim=0))
        work = dist.all_gather(tensor_list=temp,
                               tensor=tensor.transpose(0, dim).contiguous(),
                               group=gpc.get_group(parallel_mode),
                               async_op=async_op)
        out = torch.transpose(out, 0, dim)
    if async_op:
        return out, work
    else:
        return out


def reduce_scatter(tensor: Tensor,
                   dim: int,
                   parallel_mode: ParallelMode,
                   op: ReduceOp = ReduceOp.SUM,
                   async_op: bool = False) -> Tensor:
    r"""Reduces all tensors then scatters it in a specific dimension to all
    members in the parallel group.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be reduce_scattered.
        dim (int): The dimension concatenating in.
        parallel_mode (:class:`colossalai.context.ParallelMode`): Parallel group mode used in this communication.
        op (torch.distributed.ReduceOp, optional): The type of reduce operation,
            should be included in [SUM, AVG, PRODUCT, MIN, MAX, BAND, BOR, BXOR].
            More details about ReduceOp please refer to
            `ReduceOp <https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp>`_.
        async_op (bool, optional): Whether operations are asynchronous.

    Returns:
        Union[tuple(:class:`torch.Tensor`, work handle), :class:`torch.Tensor`]: The result of reduce_scatter only,
        if async_op is set to False. A tuple of output of all-gather and Async work handle, if async_op is set to True.
    """
    depth = gpc.get_world_size(parallel_mode)
    if depth == 1:
        out = tensor
        work = None
    else:
        temp = list(map(lambda x: x.contiguous(), torch.chunk(tensor, depth, dim=dim)))
        out = torch.empty(temp[0].shape, dtype=tensor.dtype, device=get_current_device())
        work = dist.reduce_scatter(output=out,
                                   input_list=temp,
                                   op=op,
                                   group=gpc.get_group(parallel_mode),
                                   async_op=async_op)
    if async_op:
        return out, work
    else:
        return out


def all_reduce(tensor: Tensor,
               parallel_mode: ParallelMode,
               op: ReduceOp = ReduceOp.SUM,
               async_op: bool = False) -> Tensor:
    r"""Reduces the tensor data across whole parallel group in such a way that all get the final result.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be all-reduced.
        parallel_mode (:class:`colossalai.context.ParallelMode`): Parallel group mode used in this communication.
        op (torch.distributed.ReduceOp, optional): The type of reduce operation,
            should be included in [SUM, AVG, PRODUCT, MIN, MAX, BAND, BOR, BXOR].
            More details about ReduceOp please refer to
            `ReduceOp <https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp>`_.
        async_op (bool, optional): Whether operations are asynchronous.

    Returns:
        Union[tuple(:class:`torch.Tensor`, work handle), :class:`torch.Tensor`]: The result of all-gather only,
        if async_op is set to False. A tuple of output of all-gather and Async work handle, if async_op is set to True.
    """
    depth = gpc.get_world_size(parallel_mode)
    if depth == 1:
        out = tensor
        work = None
    else:
        out = tensor.contiguous()
        work = dist.all_reduce(out, op=op, group=gpc.get_group(parallel_mode), async_op=async_op)
    if async_op:
        return out, work
    else:
        return out


def broadcast(tensor: Tensor, src: int, parallel_mode: ParallelMode, async_op: bool = False):
    r"""Broadcast tensors to whole parallel group. Tensor must have the same
    number of elements in all processes participating in the collective.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be broadcast.
        src (int): Source rank.
        parallel_mode (:class:`colossalai.context.ParallelMode`): Parallel group mode used in this communication.
        async_op (bool, optional): Whether operations are asynchronous.

    Returns:
        Union[tuple(:class:`torch.Tensor`, work handle), :class:`torch.Tensor`]: The tensor need to be broadcast only,
        if async_op is set to False. A tuple of output of all-gather and Async work handle, if async_op is set to True.
    """
    depth = gpc.get_world_size(parallel_mode)
    if depth == 1:
        out = tensor
        work = None
    else:
        out = tensor.contiguous()
        work = dist.broadcast(out, src=src, group=gpc.get_group(parallel_mode), async_op=async_op)
    if async_op:
        return out, work
    else:
        return out


def reduce(tensor: Tensor, dst: int, parallel_mode: ParallelMode, op: ReduceOp = ReduceOp.SUM, async_op: bool = False):
    r"""Reduce tensors across whole parallel group. Only the process with
    rank ``dst`` is going to receive the final result.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be reduced.
        dst (int): Destination rank.
        parallel_mode (:class:`colossalai.context.ParallelMode`): Parallel group mode used in this communication.
        async_op (bool, optional): Whether operations are asynchronous.

    Returns:
        Union[tuple(:class:`torch.Tensor`, work handle), :class:`torch.Tensor`]: The result of reduce only,
        if async_op is set to False. A tuple of output of all-gather and Async work handle, if async_op is set to True.
    """
    depth = gpc.get_world_size(parallel_mode)
    if depth == 1:
        out = tensor
        work = None
    else:
        out = tensor.contiguous()
        work = dist.reduce(out, dst=dst, op=op, group=gpc.get_group(parallel_mode), async_op=async_op)
    if async_op:
        return out, work
    else:
        return out
