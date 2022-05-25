#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import List, Tuple, Union
import torch
import torch.distributed as dist

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device
from functools import reduce
import operator
from .utils import split_tensor_into_1d_equal_chunks, gather_split_1d_tensor

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


def _communicate(object_send_next: Union[torch.Tensor, List[torch.Tensor]] = None,
                 object_send_prev: Union[torch.Tensor, List[torch.Tensor]] = None,
                 recv_prev: bool = False,
                 recv_next: bool = False,
                 recv_prev_shape: Union[torch.Size, List[torch.Size]] = None,
                 recv_next_shape: Union[torch.Size, List[torch.Size]] = None,
                 prev_rank: int = None,
                 next_rank: int = None,
                 dtype: torch.dtype = None,
                 scatter_gather_tensors: bool = False) -> Tuple[Union[torch.Tensor, List[torch.Tensor]]]:
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
        recv_prev_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): shape of the tensor to be received from the previous stage, defualts to None.
        recv_next_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): shape of the tensor to be received from the next stage, defualts to None.
        prev_rank (int): the rank of the previous pipeline stage, defualts to None,
        next_rank (int): the rank of the next pipeline stage, defualts to None,
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
        if isinstance(recv_prev_shape, torch.Size):
            recv_prev_chunk_shape, recv_prev_split = _get_tensor_shape(recv_prev_shape, scatter_gather_tensors)
            tensor_recv_prev = torch.empty(recv_prev_chunk_shape,
                                           requires_grad=True,
                                           device=get_current_device(),
                                           dtype=dtype)
        else:
            tensor_recv_prev = []
            for recv_shape in recv_prev_shape:
                recv_prev_chunk_shape, recv_prev_split = _get_tensor_shape(recv_shape, scatter_gather_tensors)
                tensor_recv = torch.empty(recv_prev_chunk_shape,
                                          requires_grad=True,
                                          device=get_current_device(),
                                          dtype=dtype)
                tensor_recv_prev.append(tensor_recv)
    if recv_next:
        assert recv_next_shape is not None
        if isinstance(recv_next_shape, torch.Size):
            recv_next_chunk_shape, recv_next_split = _get_tensor_shape(recv_next_shape, scatter_gather_tensors)
            tensor_recv_next = torch.empty(recv_next_chunk_shape,
                                           requires_grad=True,
                                           device=get_current_device(),
                                           dtype=dtype)
        else:
            tensor_recv_next = []
            for recv_shape in recv_next_shape:
                recv_next_chunk_shape, recv_next_split = _get_tensor_shape(recv_shape, scatter_gather_tensors)
                tensor_recv = torch.empty(recv_next_chunk_shape,
                                          requires_grad=True,
                                          device=get_current_device(),
                                          dtype=dtype)
                tensor_recv_next.append(tensor_recv)

    if object_send_prev is not None or recv_prev:
        if prev_rank is None:
            prev_rank = gpc.get_prev_global_rank(ParallelMode.PIPELINE)

    if object_send_next is not None or recv_next:
        if next_rank is None:
            next_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)

    if object_send_prev is not None:
        if isinstance(object_send_prev, torch.Tensor):
            send_prev_split = _get_tensor_shape(object_send_prev.shape, scatter_gather_tensors)[1]
        else:
            send_prev_split = _get_tensor_shape(object_send_prev[0].shape, scatter_gather_tensors)[1]
        if send_prev_split:
            if isinstance(object_send_prev, torch.Tensor):
                object_send_prev = split_tensor_into_1d_equal_chunks(object_send_prev)
            else:
                for tensor_send in object_send_prev:
                    tensor_send = split_tensor_into_1d_equal_chunks(tensor_send)

    if object_send_next is not None:
        if isinstance(object_send_next, torch.Tensor):
            send_next_split = _get_tensor_shape(object_send_next.shape, scatter_gather_tensors)[1]
        else:
            send_next_split = _get_tensor_shape(object_send_next[0].shape, scatter_gather_tensors)[1]
        if send_next_split:
            if isinstance(object_send_next, torch.Tensor):
                object_send_next = split_tensor_into_1d_equal_chunks(object_send_next)
            else:
                for tensor_send in object_send_next:
                    tensor_send = split_tensor_into_1d_equal_chunks(tensor_send)

    ops = []
    if object_send_prev is not None:
        if isinstance(object_send_prev, torch.Tensor):
            send_prev_op = dist.P2POp(dist.isend, object_send_prev, prev_rank)
            ops.append(send_prev_op)
        else:
            for tensor_send in object_send_prev:
                send_prev_op = dist.P2POp(dist.isend, tensor_send, prev_rank)
                ops.append(send_prev_op)

    if tensor_recv_prev is not None:
        if isinstance(tensor_recv_prev, torch.Tensor):
            recv_prev_op = dist.P2POp(dist.irecv, tensor_recv_prev, prev_rank)
            ops.append(recv_prev_op)
        else:
            for tensor_recv in tensor_recv_prev:
                recv_prev_op = dist.P2POp(dist.irecv, tensor_recv, prev_rank)
                ops.append(recv_prev_op)

    if tensor_recv_next is not None:
        if isinstance(tensor_recv_next, torch.Tensor):
            recv_next_op = dist.P2POp(dist.irecv, tensor_recv_next, next_rank)
            ops.append(recv_next_op)
        else:
            for tensor_recv in tensor_recv_next:
                recv_next_op = dist.P2POp(dist.irecv, tensor_recv, next_rank)
                ops.append(recv_next_op)

    if object_send_next is not None:
        if isinstance(object_send_next, torch.Tensor):
            send_next_op = dist.P2POp(dist.isend, object_send_next, next_rank)
            ops.append(send_next_op)
        else:
            for tensor_send in object_send_next:
                send_next_op = dist.P2POp(dist.isend, tensor_send, next_rank)
                ops.append(send_next_op)

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
            for tensor_recv, tensor_shape in zip(tensor_recv_prev, recv_prev_shape):
                tensor_recv = gather_split_1d_tensor(tensor_recv).view(tensor_shape).requires_grad_()

    if recv_next and recv_next_split:
        if isinstance(tensor_recv_next, torch.Tensor):
            tensor_recv_next = gather_split_1d_tensor(tensor_recv_next).view(recv_next_shape).requires_grad_()
        else:
            for tensor_recv, tensor_shape in zip(tensor_recv_next, recv_next_shape):
                tensor_recv = gather_split_1d_tensor(tensor_recv).view(tensor_shape).requires_grad_()

    return tensor_recv_prev, tensor_recv_next


def recv_forward(input_tensor_shape,
                 prev_rank=None,
                 dtype=torch.float,
                 scatter_gather_tensors=False) -> Union[torch.Tensor, List[torch.Tensor]]:
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
        input_tensor, _ = _communicate(recv_prev=True,
                                       recv_prev_shape=input_tensor_shape,
                                       prev_rank=prev_rank,
                                       dtype=dtype,
                                       scatter_gather_tensors=scatter_gather_tensors)
    return input_tensor


def recv_backward(output_grad_shape,
                  next_rank=None,
                  dtype=torch.float,
                  scatter_gather_tensors=False) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Copy the gradient tensor from the next stage in pipeline as the input gradient of this stage.

    Args:
        output_grad_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.
        next_rank (int, optional): The rank of the source of the tensor.

    Returns:
        Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: The input gradient tensor or gradident tensor list.
    """
    if gpc.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        _, output_tensor_grad = _communicate(recv_next=True,
                                             recv_next_shape=output_grad_shape,
                                             next_rank=next_rank,
                                             dtype=dtype,
                                             scatter_gather_tensors=scatter_gather_tensors)
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
        _communicate(object_send_prev=input_tensor_grad,
                     prev_rank=prev_rank,
                     scatter_gather_tensors=scatter_gather_tensors)


def send_forward_recv_backward(output_tensor,
                               output_grad_shape,
                               recv_next=True,
                               next_rank=None,
                               dtype=torch.float,
                               scatter_gather_tensors=False) -> Union[torch.Tensor, List[torch.Tensor]]:
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
        _, output_tensor_grad = _communicate(object_send_next=output_tensor,
                                             recv_next=recv_next,
                                             recv_next_shape=output_grad_shape,
                                             next_rank=next_rank,
                                             dtype=dtype,
                                             scatter_gather_tensors=scatter_gather_tensors)
    return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad,
                               input_tensor_shape,
                               recv_prev=True,
                               prev_rank=None,
                               dtype=torch.float,
                               scatter_gather_tensors=False) -> Union[torch.Tensor, List[torch.Tensor]]:
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
        input_tensor, _ = _communicate(object_send_prev=input_tensor_grad,
                                       recv_prev=recv_prev,
                                       recv_prev_shape=input_tensor_shape,
                                       prev_rank=prev_rank,
                                       dtype=dtype,
                                       scatter_gather_tensors=scatter_gather_tensors)
    return input_tensor


def send_forward_recv_forward(output_tensor,
                              input_tensor_shape,
                              recv_prev=True,
                              prev_rank=None,
                              next_rank=None,
                              dtype=torch.float,
                              scatter_gather_tensors=False) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Batched communication operation. Sends the input tensor to the 
    next stage in pipeline, while receives the output tensor from the
    previous stage in pipeline as the input of this stage.

    Args:
        output_tensor (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent.
        input_tensor_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.

    Returns:
        Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: The input tensor.
    """
    input_tensor, _ = _communicate(object_send_next=output_tensor,
                                   recv_prev=recv_prev,
                                   recv_prev_shape=input_tensor_shape,
                                   prev_rank=prev_rank,
                                   next_rank=next_rank,
                                   dtype=dtype,
                                   scatter_gather_tensors=scatter_gather_tensors)
    return input_tensor


def send_backward_recv_backward(input_tensor_grad,
                                output_grad_shape,
                                recv_next=True,
                                prev_rank=None,
                                next_rank=None,
                                dtype=torch.float,
                                scatter_gather_tensors=False) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Batched communication operation. Sends the gradient tensor to the
    previous stage in pipeline, while receives the gradient tensor from the
    next member in pipeline as the input of this stage.

    Args:
        input_tensor_grad (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent.
        output_grad_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.

    Returns:
        Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: The input gradient tensor.
    """
    _, output_tensor_grad = _communicate(object_send_prev=input_tensor_grad,
                                         recv_next=recv_next,
                                         recv_next_shape=output_grad_shape,
                                         prev_rank=prev_rank,
                                         next_rank=next_rank,
                                         dtype=dtype,
                                         scatter_gather_tensors=scatter_gather_tensors)
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
        scatter_gather_tensors=False) -> Tuple[Union[torch.Tensor, List[torch.Tensor]]]:
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
    input_tensor, output_tensor_grad = _communicate(object_send_next=output_tensor,
                                                    object_send_prev=input_tensor_grad,
                                                    recv_prev=recv_prev,
                                                    recv_next=recv_next,
                                                    recv_prev_shape=input_tensor_shape,
                                                    recv_next_shape=output_grad_shape,
                                                    prev_rank=prev_rank,
                                                    next_rank=next_rank,
                                                    dtype=dtype,
                                                    scatter_gather_tensors=scatter_gather_tensors)
    return input_tensor, output_tensor_grad
