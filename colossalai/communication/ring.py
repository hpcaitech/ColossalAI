#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device, synchronize


def ring_forward(tensor_send_next: torch.Tensor, parallel_mode: ParallelMode):
    """Sends a tensor to the next member and recieves a tensor from the previous member.
    This function returns the recieved tensor from the previous member.

    :param tensor_send_next: Tensor sent to next member
    :param parallel_mode: Parallel group mode used in this communication
    :type tensor_send_next: Tensor
    :type parallel_mode: ParallelMode
    :return: The tensor recieved from the previous
    :rtype: Tensor
    """
    buffer_shape = tensor_send_next.size()

    ops = []
    current_rank = gpc.get_global_rank()

    tensor_recv_prev = torch.empty(buffer_shape,
                                   requires_grad=True,
                                   device=get_current_device(),
                                   dtype=tensor_send_next.dtype)

    # send to next rank
    send_next_op = torch.distributed.P2POp(
        torch.distributed.isend, tensor_send_next,
        gpc.get_next_global_rank(parallel_mode))
    ops.append(send_next_op)

    # receive from prev rank
    recv_prev_op = torch.distributed.P2POp(
        torch.distributed.irecv, tensor_recv_prev,
        gpc.get_prev_global_rank(parallel_mode))
    ops.append(recv_prev_op)

    if current_rank % 2 == 0:
        ops = ops[::-1]

    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    # To protect against race condition when using batch_isend_irecv().
    synchronize()

    return tensor_recv_prev
