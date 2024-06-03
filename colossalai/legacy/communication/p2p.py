#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import operator
from functools import reduce
from typing import List, Tuple, Union

import torch
import torch.distributed as dist

from colossalai.accelerator import get_accelerator
from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.core import global_context as gpc

from .utils import gather_split_1d_tensor, split_tensor_into_1d_equal_chunks

TensorShape = Union[torch.Size, List[int], Tuple[int]]


def _get_tensor_shape(tensor_shape: TensorShape, chunk_tensor: bool = False) -> Tuple[TensorShape, bool]:
    """get the exact tensor shape when communicating and return whether the tensor is a chunk

    Args:
        tensor_shape (:class:`torch.Size`): shape of tensor
        chunk_tensor (bool, optional): whether to chunk tensor, defaults to False

    Returns:
        Tuple[Union[:class:`torch.Size`, List[int], Tuple[int]], bool]: exact tensor shape, whether to chunk tensor
    """
    if chunk_tensor:
        tensor_chunk_shape = reduce(operator.mul, tensor_shape, 1)
        tensor_parallel_world_size = gpc.get_world_size(ParallelMode.TENSOR)
        if tensor_chunk_shape % tensor_parallel_world_size == 0:
            tensor_chunk_shape = tensor_chunk_shape // tensor_parallel_world_size
        else:
            tensor_chunk_shape = tensor_shape
            chunk_tensor = False
    else:
        tensor_chunk_shape = tensor_shape
    return tensor_chunk_shape, chunk_tensor


def create_recv_buffer_with_shapes(recv_shapes, dtype, scatter_gather_tensors):
    if isinstance(recv_shapes, torch.Size):
        recv_chunk_shape, recv_split = _get_tensor_shape(recv_shapes, scatter_gather_tensors)
        buffer_recv = torch.empty(
            recv_chunk_shape, requires_grad=True, device=get_accelerator().get_current_device(), dtype=dtype
        )
        return buffer_recv, recv_split
    buffer_recv = []
    for recv_shape in recv_shapes:
        recv_chunk_shape, recv_split = _get_tensor_shape(recv_shape, scatter_gather_tensors)
        tensor_recv = torch.empty(
            recv_chunk_shape, requires_grad=True, device=get_accelerator().get_current_device(), dtype=dtype
        )
        buffer_recv.append(tensor_recv)
    return buffer_recv, recv_split


def process_object_to_send(object_send, scatter_gather_tensors):
    if isinstance(object_send, torch.Tensor):
        send_split = _get_tensor_shape(object_send.shape, scatter_gather_tensors)[1]
        if send_split:
            object_send = split_tensor_into_1d_equal_chunks(object_send)
        return object_send

    object_send_list = []
    for tensor_send in object_send:
        send_split = _get_tensor_shape(tensor_send.shape, scatter_gather_tensors)[1]
        if send_split:
            object_send_list.append(split_tensor_into_1d_equal_chunks(tensor_send))
        else:
            object_send_list.append(tensor_send)
    object_send = tuple(object_send_list)

    return object_send


def filling_ops_queue(obj, comm_op, comm_rank, ops_queue):
    if isinstance(obj, torch.Tensor):
        op_to_add = dist.P2POp(comm_op, obj, comm_rank)
        ops_queue.append(op_to_add)
    else:
        for tensor_to_comm in obj:
            op_to_add = dist.P2POp(comm_op, tensor_to_comm, comm_rank)
            ops_queue.append(op_to_add)


def _communicate(
    object_send_next: Union[torch.Tensor, List[torch.Tensor]] = None,
    object_send_prev: Union[torch.Tensor, List[torch.Tensor]] = None,
    recv_prev: bool = False,
    recv_next: bool = False,
    recv_prev_shape: Union[torch.Size, List[torch.Size]] = None,
    recv_next_shape: Union[torch.Size, List[torch.Size]] = None,
    prev_rank: int = None,
    next_rank: int = None,
    dtype: torch.dtype = None,
    scatter_gather_tensors: bool = False,
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Adapted from megatron.p2p_communication.
    Communicate tensors between stages. Used as helper method in other
    communication methods that are used in pipeline schedule.
    Takes the following arguments:
        object_send_next (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): tensor to send to next rank (no tensor sent if
                          set to None).
        object_send_prev (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev (bool): boolean for whether tensor should be received from
                   previous rank.
        recv_next (bool): boolean for whether tensor should be received from
                   next rank.
        recv_prev_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): shape of the tensor to be received from the previous stage, defaults to None.
        recv_next_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): shape of the tensor to be received from the next stage, defaults to None.
        prev_rank (int): the rank of the previous pipeline stage, defaults to None,
        next_rank (int): the rank of the next pipeline stage, defaults to None,
        dtype (torch.dtype): data type of intermediate buffers, defaults to None
        scatter_gather_tensors (bool): whether to scatter and gather tensor between pipeline stages, defaults to False

    Returns:
        Tuple[Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]]: returns tensor_recv_prev, tensor_recv_next
    """

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    if recv_prev:
        assert recv_prev_shape is not None
        tensor_recv_prev, recv_prev_split = create_recv_buffer_with_shapes(
            recv_prev_shape, dtype, scatter_gather_tensors
        )

    if recv_next:
        assert recv_next_shape is not None
        tensor_recv_next, recv_next_split = create_recv_buffer_with_shapes(
            recv_next_shape, dtype, scatter_gather_tensors
        )

    if object_send_prev is not None or recv_prev:
        if prev_rank is None:
            prev_rank = gpc.get_prev_global_rank(ParallelMode.PIPELINE)

    if object_send_next is not None or recv_next:
        if next_rank is None:
            next_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)

    if object_send_prev is not None:
        object_send_prev = process_object_to_send(object_send_prev, scatter_gather_tensors)

    if object_send_next is not None:
        object_send_next = process_object_to_send(object_send_next, scatter_gather_tensors)

    ops = []
    if object_send_prev is not None:
        filling_ops_queue(object_send_prev, dist.isend, prev_rank, ops)

    if tensor_recv_prev is not None:
        filling_ops_queue(tensor_recv_prev, dist.irecv, prev_rank, ops)

    if tensor_recv_next is not None:
        filling_ops_queue(tensor_recv_next, dist.irecv, next_rank, ops)

    if object_send_next is not None:
        filling_ops_queue(object_send_next, dist.isend, next_rank, ops)

    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()

    if recv_prev and recv_prev_split:
        if isinstance(tensor_recv_prev, torch.Tensor):
            tensor_recv_prev = gather_split_1d_tensor(tensor_recv_prev).view(recv_prev_shape).requires_grad_()
        else:
            for index in range(len(tensor_recv_prev)):
                tensor_recv_prev[index] = (
                    gather_split_1d_tensor(tensor_recv_prev[index]).view(recv_prev_shape[index]).requires_grad_()
                )

    if recv_next and recv_next_split:
        if isinstance(tensor_recv_next, torch.Tensor):
            tensor_recv_next = gather_split_1d_tensor(tensor_recv_next).view(recv_next_shape).requires_grad_()
        else:
            for index in range(len(tensor_recv_next)):
                tensor_recv_next[index] = (
                    gather_split_1d_tensor(tensor_recv_next[index]).view(recv_next_shape[index]).requires_grad_()
                )

    return tensor_recv_prev, tensor_recv_next


def recv_forward(
    input_tensor_shape, prev_rank=None, dtype=torch.float, scatter_gather_tensors=False
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Copy the forward output from the previous stage in pipeline as the input tensor of this stage.

    Args:
        input_tensor_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.
        prev_rank (int, optional): The rank of the source of the tensor.

    Returns:
        Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: The input tensor or input tensor list.
    """
    if gpc.is_pipeline_first_stage():
        input_tensor = None
    else:
        input_tensor, _ = _communicate(
            recv_prev=True,
            recv_prev_shape=input_tensor_shape,
            prev_rank=prev_rank,
            dtype=dtype,
            scatter_gather_tensors=scatter_gather_tensors,
        )
    return input_tensor


def recv_backward(
    output_grad_shape, next_rank=None, dtype=torch.float, scatter_gather_tensors=False
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Copy the gradient tensor from the next stage in pipeline as the input gradient of this stage.

    Args:
        output_grad_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.
        next_rank (int, optional): The rank of the source of the tensor.

    Returns:
        Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: The input gradient tensor or gradient tensor list.
    """
    if gpc.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        _, output_tensor_grad = _communicate(
            recv_next=True,
            recv_next_shape=output_grad_shape,
            next_rank=next_rank,
            dtype=dtype,
            scatter_gather_tensors=scatter_gather_tensors,
        )
    return output_tensor_grad


def send_forward(output_tensor, next_rank=None, scatter_gather_tensors=False) -> None:
    """Sends the input tensor to the next stage in pipeline.

    Args:
        output_tensor (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent.
        next_rank (int, optional): The rank of the recipient of the tensor.
    """
    if not gpc.is_pipeline_last_stage():
        _communicate(object_send_next=output_tensor, next_rank=next_rank, scatter_gather_tensors=scatter_gather_tensors)


def send_backward(input_tensor_grad, prev_rank=None, scatter_gather_tensors=False) -> None:
    """Sends the gradient tensor to the previous stage in pipeline.

    Args:
        input_tensor_grad (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent
        prev_rank (int, optional): The rank of the recipient of the tensor
    """
    if not gpc.is_pipeline_first_stage():
        _communicate(
            object_send_prev=input_tensor_grad, prev_rank=prev_rank, scatter_gather_tensors=scatter_gather_tensors
        )


def send_forward_recv_backward(
    output_tensor, output_grad_shape, recv_next=True, next_rank=None, dtype=torch.float, scatter_gather_tensors=False
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Batched communication operation. Sends the input tensor to the
    next stage in pipeline, while receives the gradient tensor from the
    next stage in pipeline as the input gradient tensor of this stage.

    Args:
        output_tensor (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent.
        output_grad_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.

    Returns:
        Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: The input gradient tensor.
    """
    if gpc.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        _, output_tensor_grad = _communicate(
            object_send_next=output_tensor,
            recv_next=recv_next,
            recv_next_shape=output_grad_shape,
            next_rank=next_rank,
            dtype=dtype,
            scatter_gather_tensors=scatter_gather_tensors,
        )
    return output_tensor_grad


def send_backward_recv_forward(
    input_tensor_grad,
    input_tensor_shape,
    recv_prev=True,
    prev_rank=None,
    dtype=torch.float,
    scatter_gather_tensors=False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Batched communication operation. Sends the gradient tensor to the
    previous stage in pipeline, while receives the output tensor from the
    previous stage in pipeline as the input of this stage.

    Args:
        input_tensor_grad (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent.
        input_tensor_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.

    Returns:
        Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: The input tensor.
    """
    if gpc.is_pipeline_first_stage():
        input_tensor = None
    else:
        input_tensor, _ = _communicate(
            object_send_prev=input_tensor_grad,
            recv_prev=recv_prev,
            recv_prev_shape=input_tensor_shape,
            prev_rank=prev_rank,
            dtype=dtype,
            scatter_gather_tensors=scatter_gather_tensors,
        )
    return input_tensor


def send_forward_recv_forward(
    output_tensor,
    input_tensor_shape,
    recv_prev=True,
    prev_rank=None,
    next_rank=None,
    dtype=torch.float,
    scatter_gather_tensors=False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Batched communication operation. Sends the input tensor to the
    next stage in pipeline, while receives the output tensor from the
    previous stage in pipeline as the input of this stage.

    Args:
        output_tensor (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent.
        input_tensor_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.

    Returns:
        Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: The input tensor.
    """
    input_tensor, _ = _communicate(
        object_send_next=output_tensor,
        recv_prev=recv_prev,
        recv_prev_shape=input_tensor_shape,
        prev_rank=prev_rank,
        next_rank=next_rank,
        dtype=dtype,
        scatter_gather_tensors=scatter_gather_tensors,
    )
    return input_tensor


def send_backward_recv_backward(
    input_tensor_grad,
    output_grad_shape,
    recv_next=True,
    prev_rank=None,
    next_rank=None,
    dtype=torch.float,
    scatter_gather_tensors=False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Batched communication operation. Sends the gradient tensor to the
    previous stage in pipeline, while receives the gradient tensor from the
    next member in pipeline as the input of this stage.

    Args:
        input_tensor_grad (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent.
        output_grad_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.

    Returns:
        Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: The input gradient tensor.
    """
    _, output_tensor_grad = _communicate(
        object_send_prev=input_tensor_grad,
        recv_next=recv_next,
        recv_next_shape=output_grad_shape,
        prev_rank=prev_rank,
        next_rank=next_rank,
        dtype=dtype,
        scatter_gather_tensors=scatter_gather_tensors,
    )
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
    output_tensor,
    input_tensor_grad,
    input_tensor_shape,
    output_grad_shape,
    recv_prev=True,
    recv_next=True,
    prev_rank=None,
    next_rank=None,
    dtype=torch.float,
    scatter_gather_tensors=False,
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]]]:
    """Batched communication operation. Sends the input tensor to the next stage in pipeline and
    the gradient tensor to the previous stage, while receives the input gradient tensor from the
    next stage and the input tensor from the previous stage.

    Args:
        output_tensor (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor sent to the next.
        input_tensor_grad (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor sent to the previous.
        input_tensor_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor received from the previous.
        output_grad_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor received from the next.

    Returns:
        Tuple(Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]], Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): (the input tensor, the input gradient tensor)
    """
    input_tensor, output_tensor_grad = _communicate(
        object_send_next=output_tensor,
        object_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        recv_prev_shape=input_tensor_shape,
        recv_next_shape=output_grad_shape,
        prev_rank=prev_rank,
        next_rank=next_rank,
        dtype=dtype,
        scatter_gather_tensors=scatter_gather_tensors,
    )
    return input_tensor, output_tensor_grad
