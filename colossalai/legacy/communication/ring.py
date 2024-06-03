#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch

from colossalai.accelerator import get_accelerator
from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.core import global_context as gpc


def ring_forward(tensor_send_next: torch.Tensor, parallel_mode: ParallelMode) -> torch.Tensor:
    """Sends a tensor to the next member and receives a tensor from the previous member.
    This function returns the received tensor from the previous member.

    Args:
        tensor_send_next (:class:`torch.Tensor`): Tensor sent to next member
        parallel_mode (ParallelMode): Parallel group mode used in this communication

    Returns:
        :class:`torch.Tensor`: The tensor received from the previous.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """
    buffer_shape = tensor_send_next.size()

    ops = []
    current_rank = gpc.get_global_rank()

    tensor_recv_prev = torch.empty(
        buffer_shape, requires_grad=True, device=get_accelerator().get_current_device(), dtype=tensor_send_next.dtype
    )

    # send to next rank
    send_next_op = torch.distributed.P2POp(
        torch.distributed.isend, tensor_send_next, gpc.get_next_global_rank(parallel_mode)
    )
    ops.append(send_next_op)

    # receive from prev rank
    recv_prev_op = torch.distributed.P2POp(
        torch.distributed.irecv, tensor_recv_prev, gpc.get_prev_global_rank(parallel_mode)
    )
    ops.append(recv_prev_op)

    if current_rank % 2 == 0:
        ops = ops[::-1]

    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    # To protect against race condition when using batch_isend_irecv().
    get_accelerator().synchronize()

    return tensor_recv_prev
