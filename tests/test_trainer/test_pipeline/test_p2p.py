#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from colossalai.communication import (recv_backward, recv_forward,
                                      recv_tensor_meta, send_backward,
                                      send_backward_recv_forward, send_forward,
                                      send_forward_recv_backward,
                                      send_tensor_meta)
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.initialize import launch
from colossalai.logging import get_dist_logger
from colossalai.utils import get_current_device
from functools import partial

BATCH_SIZE = 16
SEQ_LENGTH = 64
HIDDEN_SIZE = 128

CONFIG = dict(
    parallel=dict(
        pipeline=dict(size=4),
        tensor=dict(size=1, mode=None)
    ),
    seed=1024
)


def check_equal(A, B):
    return torch.allclose(A, B, rtol=1e-5, atol=1e-3)


def check_forward(output_tensor, rank, logger):
    dist.barrier()
    if gpc.is_first_rank(ParallelMode.PIPELINE):
        tensor = output_tensor.clone()
    else:
        tensor = recv_forward(output_tensor.shape)
        logger.info('Rank {} received forward. Correct tensor: {}'.format(
            rank, check_equal(tensor, output_tensor)))
    if not gpc.is_last_rank(ParallelMode.PIPELINE):
        send_forward(tensor)
        logger.info('Rank {} sent forward.'.format(rank))


def check_backward(output_grad, rank, logger):
    dist.barrier()
    if gpc.is_last_rank(ParallelMode.PIPELINE):
        grad = output_grad.clone()
    else:
        grad = recv_backward(output_grad.shape)
        logger.info('Rank {} received backward. Correct grad: {}'.format(
            rank, check_equal(grad, output_grad)))
    if not gpc.is_first_rank(ParallelMode.PIPELINE):
        send_backward(grad)
        logger.info('Rank {} sent backward.'.format(rank))


def check_forward_backward(output_tensor, output_grad, rank, logger):
    dist.barrier()
    if not gpc.is_first_rank(ParallelMode.PIPELINE):
        tensor = send_backward_recv_forward(output_grad, output_tensor.shape)
        logger.info(
            'Rank {} sent backward received forward. Correct tensor: {}'.
            format(rank, check_equal(tensor, output_tensor)))
    if not gpc.is_last_rank(ParallelMode.PIPELINE):
        grad = send_forward_recv_backward(output_tensor, output_grad.shape)
        logger.info(
            'Rank {} sent forward received backward. Correct grad: {}'.format(
                rank, check_equal(grad, output_grad)))


def check_op(size, rank, prev_rank, next_rank, up_group, down_group, logger):
    dtype = torch.float32
    device = get_current_device()
    tensor_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    # recv_tensor_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    grad_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    tensor = torch.randn(tensor_shape, dtype=dtype, device=device)
    dist.all_reduce(tensor)
    grad = torch.randn(grad_shape, dtype=dtype, device=device)
    dist.all_reduce(grad)
    if rank % 2 == 0:
        need_meta = True
        need_meta = send_tensor_meta(tensor, need_meta)
        logger.info('Rank {} shape sent (need meta: {}).'.format(
            rank, need_meta))
        req = dist.broadcast(tensor, src=rank, group=down_group, async_op=True)
        req.wait()
        out = tensor.clone()
        logger.info('Rank {} test op: tensor sent.'.format(rank))
    else:
        recv_tensor_shape = recv_tensor_meta(None)
        logger.info('Rank {} shape received. Correct shape: {}'.format(
            rank, tensor_shape == recv_tensor_shape))
        out = torch.empty(recv_tensor_shape, dtype=dtype, device=device)
        req = dist.broadcast(out, src=prev_rank, group=up_group, async_op=True)
        req.wait()
        logger.info('Rank {} test op: received tensor ({})'.format(
            rank, out.shape))

    logger.info('Rank {} test op. Correct tensor: {}'.format(
        rank, check_equal(tensor, out)))


def check_comm(size, rank, prev_rank, next_rank, up_group, down_group, logger):
    dtype = torch.float32
    device = get_current_device()
    tensor_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    grad_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    tensor = torch.randn(tensor_shape, dtype=dtype, device=device)
    dist.all_reduce(tensor)
    grad = torch.randn(grad_shape, dtype=dtype, device=device)
    dist.all_reduce(grad)
    check_op(size, rank, prev_rank, next_rank, up_group, down_group, logger)
    check_forward(tensor, rank, logger)
    check_backward(grad, rank, logger)
    check_forward_backward(tensor, grad, rank, logger)


def run_check(rank, world_size):
    launch(
        config=CONFIG,
        rank=rank,
        world_size=world_size,
        host='localhost',
        port=29932,
        backend='nccl'
    )
    logger = get_dist_logger()
    rank = gpc.get_global_rank()
    prev_rank = gpc.get_prev_global_rank(ParallelMode.PIPELINE)
    up_ranks = gpc.get_ranks_in_group(ParallelMode.PIPELINE_PREV)
    up_group = gpc.get_group(ParallelMode.PIPELINE_PREV)
    next_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)
    down_ranks = gpc.get_ranks_in_group(ParallelMode.PIPELINE_NEXT)
    down_group = gpc.get_group(ParallelMode.PIPELINE_NEXT)
    logger.info(
        'Rank {0}: prev rank {1} (up: {2}), next rank {3} (down: {4})'.format(
            rank, prev_rank, up_ranks, next_rank, down_ranks))
    logger.info('Distributed environment is initialzied.')

    check_comm(world_size, rank, prev_rank, next_rank, up_group, down_group,
               logger)
    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_p2p():
    world_size = 4
    run_func = partial(run_check, world_size=world_size)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_p2p()
