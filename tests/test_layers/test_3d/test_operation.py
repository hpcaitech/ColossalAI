#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from colossalai.context import ParallelMode
from colossalai.core import global_context
from colossalai.logging import get_dist_logger
from colossalai.nn.layer.parallel_3d._operation import *
from colossalai.utils import get_current_device

from common import *


def check_AB():
    rank = torch.distributed.get_rank()
    logger = get_dist_logger()
    dtype = torch.float
    j = global_context.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
    i = global_context.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
    k = global_context.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[k]
    A = torch.chunk(A, DEPTH, dim=0)[j]
    A = A.clone()
    A.requires_grad = True

    B_shape = (HIDDEN_SIZE, 4 * HIDDEN_SIZE)
    B_master = torch.randn(B_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(B_master, src=0)
    B = torch.chunk(B_master, DEPTH, dim=0)[k]
    B = torch.chunk(B, DEPTH, dim=-1)[j]
    B = torch.chunk(B, DEPTH, dim=-1)[i]
    B = B.clone()
    B.requires_grad = True

    out = Matmul_AB_3D.apply(A, B, DEPTH, ParallelMode.PARALLEL_3D_INPUT,
                             ParallelMode.PARALLEL_3D_WEIGHT,
                             ParallelMode.PARALLEL_3D_OUTPUT)

    C_shape = (BATCH_SIZE, SEQ_LENGTH, 4 * HIDDEN_SIZE)
    A_master = A_master.clone()
    A_master.requires_grad = True
    B_master = B_master.clone()
    B_master.requires_grad = True
    C_master = torch.matmul(A_master, B_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]
    C = torch.chunk(C, DEPTH, dim=0)[k]
    # check forward correctness
    logger.info('Rank {} AB forward: {}'.format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape,
                              dtype=dtype,
                              device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    grad = torch.chunk(grad, DEPTH, dim=0)[k]

    out.backward(grad)

    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[k]
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[j]
    # check backward correctness
    logger.info('Rank {} AB backward (A_grad): {}'.format(
        rank, check_equal(A_grad, A.grad)))

    B_grad = B_master.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[k]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[j]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[i]
    # check backward correctness
    logger.info('Rank {} AB backward (B_grad): {}'.format(
        rank, check_equal(B_grad, B.grad)))


def check_ABT():
    rank = torch.distributed.get_rank()
    logger = get_dist_logger()
    dtype = torch.float

    j = A_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
    i = B_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
    k = C_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)
    device = get_current_device()

    C_shape = (BATCH_SIZE, SEQ_LENGTH, 4 * HIDDEN_SIZE)
    C_master = torch.randn(C_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(C_master, src=0)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]
    C = torch.chunk(C, DEPTH, dim=0)[k]
    C = C.clone()
    C.requires_grad = True

    B_shape = (HIDDEN_SIZE, 4 * HIDDEN_SIZE)
    B_master = torch.randn(B_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(B_master, src=0)
    B = torch.chunk(B_master, DEPTH, dim=0)[k]
    B = torch.chunk(B, DEPTH, dim=-1)[j]
    B = torch.chunk(B, DEPTH, dim=-1)[i]
    B = B.clone()
    B.requires_grad = True

    out = Matmul_ABT_3D.apply(C, B, DEPTH, ParallelMode.PARALLEL_3D_OUTPUT,
                              ParallelMode.PARALLEL_3D_WEIGHT,
                              ParallelMode.PARALLEL_3D_INPUT)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    C_master = C_master.clone()
    C_master.requires_grad = True
    B_master = B_master.clone()
    B_master.requires_grad = True
    A_master = torch.matmul(C_master, B_master.transpose(0, 1))
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[k]
    A = torch.chunk(A, DEPTH, dim=0)[j]
    logger.info('Rank {} ABT forward: {}'.format(rank, check_equal(out, A)))

    grad_shape = A_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[k]
    grad = torch.chunk(grad, DEPTH, dim=0)[j]

    # backward
    out.backward(grad)

    A_master.backward(grad_master)
    C_grad = C_master.grad
    C_grad = torch.chunk(C_grad, DEPTH, dim=0)[i]
    C_grad = torch.chunk(C_grad, DEPTH, dim=-1)[j]
    C_grad = torch.chunk(C_grad, DEPTH, dim=0)[k]
    logger.info('Rank {} ABT backward (A_grad): {}'.format(
        rank, check_equal(C_grad, C.grad)))

    B_grad = B_master.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[k]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[j]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[i]
    logger.info('Rank {} ABT backward (B_grad): {}'.format(
        rank, check_equal(B_grad, B.grad)))


def check_ATB():
    rank = torch.distributed.get_rank()
    logger = get_dist_logger()
    device = get_current_device()
    dtype = torch.float

    j = A_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
    i = B_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
    k = C_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[k]
    A = torch.chunk(A, DEPTH, dim=0)[j]
    A = A.clone()
    A.requires_grad = True

    C_shape = (BATCH_SIZE, SEQ_LENGTH, 4 * HIDDEN_SIZE)
    C_master = torch.randn(C_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(C_master, src=0)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]
    C = torch.chunk(C, DEPTH, dim=0)[k]
    C = C.clone()
    C.requires_grad = True

    out = Matmul_ATB_3D.apply(A, C, DEPTH, ParallelMode.PARALLEL_3D_INPUT,
                              ParallelMode.PARALLEL_3D_OUTPUT,
                              ParallelMode.PARALLEL_3D_WEIGHT)

    B_shape = (HIDDEN_SIZE, 4 * HIDDEN_SIZE)
    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = C_master.clone()
    C_master.requires_grad = True
    B_master = torch.matmul(
        A_master.view(-1, A_master.shape[-1]).transpose(0, 1),
        C_master.view(-1, C_master.shape[-1]))
    B = torch.chunk(B_master, DEPTH, dim=0)[k]
    B = torch.chunk(B, DEPTH, dim=-1)[j]
    B = torch.chunk(B, DEPTH, dim=-1)[i]
    logger.info('Rank {} ATB forward: {}'.format(rank, check_equal(out, B)))

    grad_shape = B_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[k]
    grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    grad = torch.chunk(grad, DEPTH, dim=-1)[i]

    out.backward(grad)

    B_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[k]
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[j]
    logger.info('Rank {} ATB backward (A_grad): {}'.format(
        rank, check_equal(A_grad, A.grad)))

    C_grad = C_master.grad
    C_grad = torch.chunk(C_grad, DEPTH, dim=0)[i]
    C_grad = torch.chunk(C_grad, DEPTH, dim=-1)[j]
    C_grad = torch.chunk(C_grad, DEPTH, dim=0)[k]
    logger.info('Rank {} ATB backward (B_grad): {}'.format(
        rank, check_equal(C_grad, C.grad)))


def check_add():
    rank = torch.distributed.get_rank()
    logger = get_dist_logger()
    dtype = torch.float

    j = A_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
    i = B_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
    k = C_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)
    device = get_current_device()

    A_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[k]
    A = torch.chunk(A, DEPTH, dim=0)[j]
    A = A.clone()
    A.requires_grad = True

    bias_shape = (HIDDEN_SIZE, )
    bias_master = torch.randn(bias_shape,
                              dtype=dtype,
                              device=get_current_device())
    torch.distributed.broadcast(bias_master, src=0)
    bias = torch.chunk(bias_master, DEPTH)[j]
    bias = torch.chunk(bias, DEPTH)[i]
    bias = bias.clone()
    bias.requires_grad = True

    out = Add_3D.apply(A, bias, DEPTH, ParallelMode.PARALLEL_3D_INPUT,
                       ParallelMode.PARALLEL_3D_WEIGHT,
                       ParallelMode.PARALLEL_3D_OUTPUT)

    A_master = A_master.clone()
    A_master.requires_grad = True
    bias_master = bias_master.clone()
    bias_master.requires_grad = True
    C_master = A_master + bias_master
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[k]
    C = torch.chunk(C, DEPTH, dim=0)[j]

    logger.info('Rank {} Add forward: {}'.format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[k]
    grad = torch.chunk(grad, DEPTH, dim=0)[j]

    out.backward(grad)

    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[k]
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[j]
    logger.info('Rank {} Add backward (A_grad): {}'.format(
        rank, check_equal(A_grad, A.grad)))

    if j == k:
        bias_grad = bias_master.grad
        bias_grad = torch.chunk(bias_grad, DEPTH)[j]
        bias_grad = torch.chunk(bias_grad, DEPTH)[i]
        logger.info('Rank {} Add backward (b_grad): {}'.format(
            rank, check_equal(bias_grad, bias.grad)))
    else:
        logger.info('Rank {} Add backward (b_grad): {}'.format(
            rank,
            # np.count_nonzero(bias.grad.detach().cpu().numpy()) == 0))
            bias.grad is None))


def check_mul():
    rank = torch.distributed.get_rank()
    logger = get_dist_logger()
    dtype = torch.float

    j = A_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
    i = B_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
    k = C_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)
    device = get_current_device()

    A_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[k]
    A = torch.chunk(A, DEPTH, dim=0)[j]
    A = A.clone()
    A.requires_grad = True

    bias_shape = (HIDDEN_SIZE, )
    bias_master = torch.randn(bias_shape,
                              dtype=dtype,
                              device=get_current_device())
    torch.distributed.broadcast(bias_master, src=0)
    bias = torch.chunk(bias_master, DEPTH)[j]
    bias = torch.chunk(bias, DEPTH)[i]
    bias = bias.clone()
    bias.requires_grad = True

    out = Mul_3D.apply(A, bias, DEPTH, ParallelMode.PARALLEL_3D_INPUT,
                       ParallelMode.PARALLEL_3D_WEIGHT,
                       ParallelMode.PARALLEL_3D_OUTPUT)

    A_master = A_master.clone()
    A_master.requires_grad = True
    bias_master = bias_master.clone()
    bias_master.requires_grad = True
    C_master = torch.mul(A_master, bias_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[k]
    C = torch.chunk(C, DEPTH, dim=0)[j]

    logger.info('Rank {} Mul forward: {}'.format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[k]
    grad = torch.chunk(grad, DEPTH, dim=0)[j]

    out.backward(grad)

    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[k]
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[j]
    logger.info('Rank {} Mul backward (A_grad): {}'.format(
        rank, check_equal(A_grad, A.grad)))

    if j == k:
        bias_grad = bias_master.grad
        bias_grad = torch.chunk(bias_grad, DEPTH)[j]
        bias_grad = torch.chunk(bias_grad, DEPTH)[i]
        logger.info('Rank {} Mul backward (b_grad): {}'.format(
            rank, check_equal(bias_grad, bias.grad)))
    else:
        logger.info('Rank {} Mul backward (b_grad): {}'.format(
            rank,
            # np.count_nonzero(bias.grad.detach().cpu().numpy()) == 0))
            bias.grad is None))


def check_sum():
    rank = torch.distributed.get_rank()
    logger = get_dist_logger()
    dtype = torch.float

    j = A_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
    i = B_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
    k = C_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)
    device = get_current_device()

    # tensor
    A_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[k]
    A = torch.chunk(A, DEPTH, dim=0)[j]
    A = A.clone()
    A.requires_grad = True

    out_tensor = Sum_3D.apply(A, -1, DEPTH, ParallelMode.PARALLEL_3D_OUTPUT)

    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = torch.sum(A_master, dim=-1)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=0)[j]
    logger.info('Rank {} Sum forward: {}'.format(rank,
                                                 check_equal(out_tensor, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=0)[j]

    out_tensor.backward(grad / DEPTH)

    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[k]
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[j]
    logger.info('Rank {} Sum backward: {}'.format(rank,
                                                  check_equal(A_grad, A.grad)))


def check_reduce():
    rank = torch.distributed.get_rank()
    logger = get_dist_logger()
    dtype = torch.float

    j = A_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
    i = B_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
    k = C_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)
    device = get_current_device()

    # scaler
    B_shape = (DEPTH * DEPTH, DEPTH)
    B_master = torch.randn(B_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(B_master, src=0)
    B = torch.chunk(B_master, DEPTH, dim=0)[i]
    B = torch.chunk(B, DEPTH, dim=-1)[k]
    B = torch.chunk(B, DEPTH, dim=0)[j]
    B = torch.squeeze(B)
    B = B.clone()
    B.requires_grad = True

    out_scaler = Reduce_3D.apply(B, 0, DEPTH, ParallelMode.PARALLEL_3D_OUTPUT)
    out_scaler = Reduce_3D.apply(out_scaler, 0, DEPTH,
                                 ParallelMode.PARALLEL_3D_INPUT)
    out_scaler = Reduce_3D.apply(out_scaler, 0, DEPTH,
                                 ParallelMode.PARALLEL_3D_WEIGHT)

    B_master = B_master.clone()
    B_master.requires_grad = True
    D = torch.sum(B_master)
    logger.info('Rank {} Reduce forward: {}'.format(rank,
                                                    check_equal(out_scaler,
                                                                D)))

    grad_shape = D.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)

    out_scaler.backward(grad_master)

    D.backward(grad_master)
    B_grad = B_master.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[i]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[k]
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[j]
    B_grad = torch.squeeze(B_grad)
    logger.info('Rank {} Reduce backward: {}'.format(
        rank, check_equal(B_grad, B.grad)))
