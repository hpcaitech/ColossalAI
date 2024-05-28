#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest
import torch
import torch.distributed as dist

from colossalai.accelerator import get_accelerator
from colossalai.legacy.communication import (
    recv_backward,
    recv_forward,
    send_backward,
    send_backward_recv_forward,
    send_forward,
    send_forward_recv_backward,
)
from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.initialize import launch
from colossalai.logging import get_dist_logger
from colossalai.testing import rerun_if_address_is_in_use, spawn

BATCH_SIZE = 4
SEQ_LENGTH = 2
HIDDEN_SIZE = 16

CONFIG = dict(parallel=dict(pipeline=dict(size=4), tensor=dict(size=1, mode=None)), seed=1024)


def check_equal(A, B):
    return torch.allclose(A, B, rtol=1e-5, atol=1e-3)


def check_forward(output_tensor, rank, logger):
    dist.barrier()
    if gpc.is_first_rank(ParallelMode.PIPELINE):
        tensor = output_tensor.clone()
    else:
        tensor = recv_forward(output_tensor.shape)
        logger.info("Rank {} received forward. Correct tensor: {}".format(rank, check_equal(tensor, output_tensor)))
    if not gpc.is_last_rank(ParallelMode.PIPELINE):
        send_forward(tensor)
        logger.info("Rank {} sent forward.".format(rank))


def check_backward(output_grad, rank, logger):
    dist.barrier()
    if gpc.is_last_rank(ParallelMode.PIPELINE):
        grad = output_grad.clone()
    else:
        grad = recv_backward(output_grad.shape)
        logger.info("Rank {} received backward. Correct grad: {}".format(rank, check_equal(grad, output_grad)))
    if not gpc.is_first_rank(ParallelMode.PIPELINE):
        send_backward(grad)
        logger.info("Rank {} sent backward.".format(rank))


def check_forward_backward(output_tensor, output_grad, rank, logger):
    dist.barrier()
    if not gpc.is_first_rank(ParallelMode.PIPELINE):
        tensor = send_backward_recv_forward(output_grad, output_tensor.shape)
        logger.info(
            "Rank {} sent backward received forward. Correct tensor: {}".format(
                rank, check_equal(tensor, output_tensor)
            )
        )
    if not gpc.is_last_rank(ParallelMode.PIPELINE):
        grad = send_forward_recv_backward(output_tensor, output_grad.shape)
        logger.info(
            "Rank {} sent forward received backward. Correct grad: {}".format(rank, check_equal(grad, output_grad))
        )


def check_comm(size, rank, prev_rank, next_rank, logger):
    dtype = torch.float32
    device = get_accelerator().get_current_device()
    tensor_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    grad_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    tensor = torch.randn(tensor_shape, dtype=dtype, device=device)
    dist.all_reduce(tensor)
    grad = torch.randn(grad_shape, dtype=dtype, device=device)
    dist.all_reduce(grad)
    check_forward(tensor, rank, logger)
    check_backward(grad, rank, logger)
    check_forward_backward(tensor, grad, rank, logger)


def run_check(rank, world_size, port):
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    logger = get_dist_logger()
    rank = gpc.get_global_rank()
    prev_rank = gpc.get_prev_global_rank(ParallelMode.PIPELINE)
    next_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)
    logger.info("Rank {0}: prev rank {1}, next rank {2}".format(rank, prev_rank, next_rank))
    logger.info("Distributed environment is initialized.")

    check_comm(world_size, rank, prev_rank, next_rank, logger)
    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_p2p():
    world_size = 4
    spawn(run_check, world_size)


if __name__ == "__main__":
    test_p2p()
