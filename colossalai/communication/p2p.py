#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.distributed as dist

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device


def _communicate(tensor_send_next=None,
                 tensor_send_prev=None,
                 recv_prev=False,
                 recv_next=False,
                 recv_prev_shape=None,
                 recv_next_shape=None,
                 prev_rank=None,
                 next_rank=None,
                 up_group=None,
                 down_group=None,
                 dtype=None):
    """
    Adapted from megatron.p2p_communication.
    Communicate tensors between stages. Used as helper method in other
    communication methods that are used in pipeline schedule.
    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
    Returns:
        (tensor_recv_prev, tensor_recv_next)
    """

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    if recv_prev:
        assert recv_prev_shape is not None
        tensor_recv_prev = torch.empty(recv_prev_shape,
                                       requires_grad=True,
                                       device=get_current_device(),
                                       dtype=dtype)
    if recv_next:
        assert recv_next_shape is not None
        tensor_recv_next = torch.empty(recv_next_shape,
                                       requires_grad=True,
                                       device=get_current_device(),
                                       dtype=dtype)

    if tensor_send_prev is not None or recv_prev:
        if prev_rank is None:
            prev_rank = gpc.get_prev_global_rank(
                ParallelMode.PIPELINE)
        if up_group is None:
            up_group = gpc.get_group(ParallelMode.PIPELINE_PREV)

    if tensor_send_next is not None or recv_next:
        if next_rank is None:
            next_rank = gpc.get_next_global_rank(
                ParallelMode.PIPELINE)
        if down_group is None:
            down_group = gpc.get_group(ParallelMode.PIPELINE_NEXT)

    # rank = dist.get_rank()
    rank = gpc.get_global_rank()

    ops = []
    if tensor_send_prev is not None:
        send_prev_op = dist.broadcast(tensor_send_prev,
                                      src=rank,
                                      group=up_group,
                                      async_op=True)
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = dist.broadcast(tensor_recv_prev,
                                      src=prev_rank,
                                      group=up_group,
                                      async_op=True)
        ops.append(recv_prev_op)
    if tensor_recv_next is not None:
        recv_next_op = dist.broadcast(tensor_recv_next,
                                      src=next_rank,
                                      group=down_group,
                                      async_op=True)
        ops.append(recv_next_op)
    if tensor_send_next is not None:
        send_next_op = dist.broadcast(tensor_send_next,
                                      src=rank,
                                      group=down_group,
                                      async_op=True)
        ops.append(send_next_op)
    for req in ops:
        req.wait()
    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()
    return tensor_recv_prev, tensor_recv_next


def recv_forward(input_tensor_shape, prev_rank=None, up_group=None):
    """Receives the input tensor from the previous member in pipeline.
    
    :param input_tensor_shape: The shape of the tensor to be recieved
    :param prev_rank: The rank of the source of the tensor
    :param up_group: Communication group including the previous member in pipeline parallel group
    :type input_tensor_shape: torch.Size
    :type prev_rank: int, optional
    :type up_group: ProcessGroup, optional
    :return: The input tensor in forward step
    :rtype: Tensor
    """
    if gpc.is_first_rank(ParallelMode.PIPELINE):
        input_tensor = None
    else:
        input_tensor, _ = _communicate(recv_prev=True,
                                       recv_prev_shape=input_tensor_shape,
                                       prev_rank=prev_rank,
                                       up_group=up_group)
    return input_tensor


def recv_backward(output_grad_shape, next_rank=None, down_group=None):
    """Receives the grad tensor from the next member in pipeline.
    
    :param output_grad_shape: The shape of the tensor to be recieved
    :param next_rank: The rank of the source of the tensor
    :param down_group: Communication group including the next member in pipeline parallel group
    :type output_grad_shape: torch.Size
    :type next_rank: int, optional
    :type down_group: ProcessGroup, optional
    :return: The grad of output tensor in forward step
    :rtype: Tensor
    """
    if gpc.is_last_rank(ParallelMode.PIPELINE):
        output_tensor_grad = None
    else:
        _, output_tensor_grad = _communicate(recv_next=True,
                                             recv_next_shape=output_grad_shape,
                                             next_rank=next_rank,
                                             down_group=down_group)
    return output_tensor_grad


def send_forward(output_tensor,
                 next_rank=None,
                 down_group=None):
    """Sends the input tensor to the next member in pipeline.
    
    :param output_tensor: Tensor to be sent
    :param next_rank: The rank of the recipient of the tensor
    :param down_group: Communication group including the next member in pipeline parallel group
    :type output_tensor: Tensor
    :type next_rank: int, optional
    :type down_group: ProcessGroup, optional
    """
    if not gpc.is_last_rank(ParallelMode.PIPELINE):
        _communicate(tensor_send_next=output_tensor,
                     next_rank=next_rank,
                     down_group=down_group)


def send_backward(input_tensor_grad,
                  prev_rank=None,
                  up_group=None):
    """Sends the grad tensor to the previous member in pipeline.
    
    :param input_tensor_grad: Tensor to be sent
    :param prev_rank: The rank of the recipient of the tensor
    :param up_group: Communication group including the previous member in pipeline parallel group
    :type input_tensor_grad: Tensor
    :type prev_rank: int, optional
    :type up_group: ProcessGroup, optional
    """
    if not gpc.is_first_rank(ParallelMode.PIPELINE):
        _communicate(tensor_send_prev=input_tensor_grad,
                     prev_rank=prev_rank,
                     up_group=up_group)


def send_forward_recv_backward(output_tensor,
                               output_grad_shape,
                               recv_next=True,
                               next_rank=None,
                               down_group=None):
    """Batched communication operation. Sends the input tensor to the 
    next member in pipeline, while recieves the grad tensor from the
    next member in pipeline.
    
    :param output_tensor: Tensor to be sent
    :param output_grad_shape: The shape of the tensor to be recieved
    :type output_tensor: Tensor
    :type output_grad_shape: torch.Size
    :return: The grad of output tensor in forward step
    :rtype: Tensor
    """
    if gpc.is_last_rank(ParallelMode.PIPELINE):
        output_tensor_grad = None
    else:
        _, output_tensor_grad = _communicate(tensor_send_next=output_tensor,
                                             recv_next=recv_next,
                                             recv_next_shape=output_grad_shape,
                                             next_rank=next_rank,
                                             down_group=down_group)
    return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad,
                               input_tensor_shape,
                               recv_prev=True,
                               prev_rank=None,
                               up_group=None):
    """Batched communication operation. Sends the grad tensor to the 
    previous member in pipeline, while recieves the input tensor from the
    previous member in pipeline.
    
    :param input_tensor_grad: Tensor to be sent
    :param input_tensor_shape: The shape of the tensor to be recieved
    :type input_tensor_grad: Tensor
    :type input_tensor_shape: torch.Size
    :return: The input tensor in forward step
    :rtype: Tensor
    """
    if gpc.is_first_rank(ParallelMode.PIPELINE):
        input_tensor = None
    else:
        input_tensor, _ = _communicate(tensor_send_prev=input_tensor_grad,
                                       recv_prev=recv_prev,
                                       recv_prev_shape=input_tensor_shape,
                                       prev_rank=prev_rank,
                                       up_group=up_group)
    return input_tensor


def send_forward_recv_forward(output_tensor,
                              input_tensor_shape,
                              recv_prev=True,
                              prev_rank=None,
                              next_rank=None,
                              up_group=None,
                              down_group=None):
    """Batched communication operation. Sends the input tensor to the 
    next member in pipeline, while recieves the input tensor from the
    previous member in pipeline.
    
    :param output_tensor: Tensor to be sent
    :param input_tensor_shape: The shape of the tensor to be recieved
    :type output_tensor: Tensor
    :type input_tensor_shape: torch.Size
    :return: The input tensor in forward step
    :rtype: Tensor
    """
    input_tensor, _ = _communicate(tensor_send_next=output_tensor,
                                   recv_prev=recv_prev,
                                   recv_prev_shape=input_tensor_shape,
                                   prev_rank=prev_rank,
                                   next_rank=next_rank,
                                   up_group=up_group,
                                   down_group=down_group)
    return input_tensor


def send_backward_recv_backward(input_tensor_grad,
                                output_grad_shape,
                                recv_next=True,
                                prev_rank=None,
                                next_rank=None,
                                up_group=None,
                                down_group=None):
    """Batched communication operation. Sends the grad tensor to the 
    previous member in pipeline, while recieves the grad tensor from the
    next member in pipeline.
    
    :param input_tensor_grad: Tensor to be sent
    :param output_grad_shape: The shape of the tensor to be recieved
    :type input_tensor_grad: Tensor
    :type output_grad_shape: torch.Size
    :return: The grad of output tensor in forward step
    :rtype: Tensor
    """
    _, output_tensor_grad = _communicate(tensor_send_prev=input_tensor_grad,
                                         recv_next=recv_next,
                                         recv_next_shape=output_grad_shape,
                                         prev_rank=prev_rank,
                                         next_rank=next_rank,
                                         up_group=up_group,
                                         down_group=down_group)
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(output_tensor,
                                                input_tensor_grad,
                                                input_tensor_shape,
                                                output_grad_shape,
                                                recv_prev=True,
                                                recv_next=True,
                                                prev_rank=None,
                                                next_rank=None,
                                                up_group=None,
                                                down_group=None):
    """Batched communication operation. Sends the input tensor to the next and 
    the grad tensor to the previous, while recieves the grad tensor from the
    next and the input tensor from the previous.
    
    :param output_tensor: Tensor sent to the next
    :param input_tensor_grad: Tensor sent to the previous
    :param input_tensor_shape: The shape of the tensor recieved from the previous
    :param output_grad_shape: The shape of the tensor recieved from the next
    :type output_tensor: Tensor
    :type input_tensor_grad: Tensor
    :type input_tensor_shape: torch.Size
    :type output_grad_shape: torch.Size
    :return: (the input tensor in forward step, the grad of output tensor in forward step)
    :rtype: (Tensor, Tensor)
    """
    input_tensor, output_tensor_grad = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        recv_prev_shape=input_tensor_shape,
        recv_next_shape=output_grad_shape,
        prev_rank=prev_rank,
        next_rank=next_rank,
        up_group=up_group,
        down_group=down_group)
    return input_tensor, output_tensor_grad
