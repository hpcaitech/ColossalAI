#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ReduceOp

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc

_all_gather_func = dist._all_gather_base \
    if "all_gather_into_tensor" not in dir(dist) else dist.all_gather_into_tensor
_reduce_scatter_func = dist._reduce_scatter_base \
    if "reduce_scatter_tensor" not in dir(dist) else dist.reduce_scatter_tensor


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
        tensor_in = tensor.contiguous() if dim == 0 else tensor.transpose(0, dim).contiguous()
        out_shape = (tensor_in.shape[0] * depth,) + tensor_in.shape[1:]
        tensor_out = torch.empty(out_shape, dtype=tensor.dtype, device=tensor.device)
        group = gpc.get_cpu_group(parallel_mode) if tensor.device.type == "cpu" else gpc.get_group(parallel_mode)
        work = _all_gather_func(tensor_out, tensor_in, group=group, async_op=async_op)
        out = tensor_out if dim == 0 else tensor_out.transpose(0, dim)
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
        tensor_in = tensor.contiguous() if dim == 0 else tensor.transpose(0, dim).contiguous()
        out_shape = (tensor_in.shape[0] // depth,) + tensor_in.shape[1:]
        tensor_out = torch.empty(out_shape, dtype=tensor.dtype, device=tensor.device)
        group = gpc.get_cpu_group(parallel_mode) if tensor.device.type == "cpu" else gpc.get_group(parallel_mode)
        work = _reduce_scatter_func(tensor_out, tensor_in, op=op, group=group, async_op=async_op)
        out = tensor_out if dim == 0 else tensor_out.transpose(0, dim)
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
        group = gpc.get_cpu_group(parallel_mode) if tensor.device.type == "cpu" else gpc.get_group(parallel_mode)
        work = dist.all_reduce(out, op=op, group=group, async_op=async_op)
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
        group = gpc.get_cpu_group(parallel_mode) if tensor.device.type == "cpu" else gpc.get_group(parallel_mode)
        work = dist.broadcast(out, src=src, group=group, async_op=async_op)
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
        group = gpc.get_cpu_group(parallel_mode) if tensor.device.type == "cpu" else gpc.get_group(parallel_mode)
        work = dist.reduce(out, dst=dst, op=op, group=group, async_op=async_op)
    if async_op:
        return out, work
    else:
        return out


def scatter_object_list(scatter_object_output_list, scatter_object_input_list, src=0, group=None) -> None:
    r"""Modified from `torch.distributed.scatter_object_list
    <https://pytorch.org/docs/stable/_modules/torch/distributed/distributed_c10d.html#scatter_object_list>` to fix issues
    """
    if dist.distributed_c10d._rank_not_in_group(group):
        return

    if (not isinstance(scatter_object_output_list, list) or len(scatter_object_output_list) < 1):
        raise RuntimeError("Expected argument scatter_object_output_list to be a list of size at least 1.")

    # set tensor device to cuda if backend is nccl
    device = torch.cuda.current_device() if dist.get_backend(group) == 'nccl' else torch.device("cpu")

    my_rank = dist.get_rank()    # use global rank
    if my_rank == src:
        tensor_list, tensor_sizes = zip(
            *[dist.distributed_c10d._object_to_tensor(obj) for obj in scatter_object_input_list])
        tensor_list = list(map(lambda x: x.to(device), tensor_list))
        tensor_sizes = list(map(lambda x: x.to(device), tensor_sizes))

    # Src rank broadcasts the maximum tensor size. This is because all ranks are
    # expected to call into scatter() with equal-sized tensors.
    if my_rank == src:
        max_tensor_size = max(tensor_sizes)
        for tensor in tensor_list:
            tensor.resize_(max_tensor_size)
    else:
        max_tensor_size = torch.tensor([0], dtype=torch.long).to(device)

    dist.broadcast(max_tensor_size, src=src, group=group)

    # Scatter actual serialized objects
    output_tensor = torch.empty(max_tensor_size.item(), dtype=torch.uint8).to(device)
    dist.scatter(
        output_tensor,
        scatter_list=None if my_rank != src else tensor_list,
        src=src,
        group=group,
    )

    # Scatter per-object sizes to trim tensors when deserializing back to object
    obj_tensor_size = torch.tensor([0], dtype=torch.long).to(device)
    dist.scatter(
        obj_tensor_size,
        scatter_list=None if my_rank != src else tensor_sizes,
        src=src,
        group=group,
    )

    output_tensor, obj_tensor_size = output_tensor.cpu(), obj_tensor_size.cpu()
    # Deserialize back to object
    scatter_object_output_list[0] = dist.distributed_c10d._tensor_to_object(output_tensor, obj_tensor_size)
