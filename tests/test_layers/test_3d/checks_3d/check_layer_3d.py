#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time

from colossalai.constants import (INPUT_GROUP_3D, OUTPUT_GROUP_3D, WEIGHT_GROUP_3D)
from colossalai.core import global_context
from colossalai.logging import get_dist_logger
from colossalai.nn import (Classifier3D, CrossEntropyLoss3D, LayerNorm3D, Linear3D, PatchEmbedding3D, VanillaClassifier,
                           VanillaPatchEmbedding)
from colossalai.nn.layer.parallel_3d._utils import get_parallel_mode_from_env
from colossalai.utils import get_current_device, print_rank_0

from .common import *
import torch


def check_linear():
    rank = torch.distributed.get_rank()
    logger = get_dist_logger()
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = 2 * HIDDEN_SIZE

    input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
    weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
    output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)

    j = A_rank = global_context.get_local_rank(input_parallel_mode)
    i = B_rank = global_context.get_local_rank(weight_parallel_mode)
    k = C_rank = global_context.get_local_rank(output_parallel_mode)

    layer = Linear3D(INPUT_SIZE, OUTPUT_SIZE, dtype=dtype, bias=True)
    layer = layer.to(device)
    layer_master = torch.nn.Linear(INPUT_SIZE, OUTPUT_SIZE)
    layer_master = layer_master.to(device)

    weight_master = layer_master.weight.data.transpose(0, 1)
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=0)[k]
    weight = torch.chunk(weight, DEPTH, dim=-1)[j]
    layer.weight.data.copy_(weight)
    bias_master = layer_master.bias.data
    torch.distributed.broadcast(bias_master, src=0)
    bias = torch.chunk(bias_master, DEPTH)[j]
    layer.bias.data.copy_(bias)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[k]
    A = torch.chunk(A, DEPTH, dim=0)[j]
    A = A.clone()
    A.requires_grad = True

    fwd_start = time.time()
    out = layer(A)
    torch.cuda.synchronize()
    fwd_end = time.time()
    print_rank_0(
        'linear forward: {0} --> {1} | {2:.3f} s'.format(tuple(A.shape), tuple(out.shape), fwd_end - fwd_start), logger)
    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = layer_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]
    C = torch.chunk(C, DEPTH, dim=0)[k]
    logger.info('Rank {} linear forward: {}'.format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    grad = torch.chunk(grad, DEPTH, dim=0)[k]

    bwd_start = time.time()
    out.backward(grad)
    torch.cuda.synchronize()
    bwd_end = time.time()
    print_rank_0('linear backward: {:.3f} s'.format(bwd_end - bwd_start), logger)

    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[k]
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[j]
    logger.info('Rank {} linear backward (input_grad): {}'.format(rank, check_equal(A_grad, A.grad)))

    B_grad = layer_master.weight.grad.transpose(0, 1)
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[k]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[j]
    logger.info('Rank {} linear backward (weight_grad): {}'.format(rank, check_equal(B_grad, layer.weight.grad)))

    bias_grad = layer_master.bias.grad
    bias_grad = torch.chunk(bias_grad, DEPTH)[j]
    logger.info('Rank {} linear backward (bias_grad): {}'.format(rank, check_equal(bias_grad, layer.bias.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_layernorm():
    rank = torch.distributed.get_rank()
    logger = get_dist_logger()
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE

    input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
    weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
    output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)

    j = A_rank = global_context.get_local_rank(input_parallel_mode)
    i = B_rank = global_context.get_local_rank(weight_parallel_mode)
    k = C_rank = global_context.get_local_rank(output_parallel_mode)

    norm = LayerNorm3D(INPUT_SIZE, eps=1e-6, dtype=dtype)
    norm = norm.to(device)
    norm_master = torch.nn.LayerNorm(INPUT_SIZE, eps=1e-6)
    norm_master = norm_master.to(device)

    weight_master = norm_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH)[k]
    norm.weight.data.copy_(weight)
    bias_master = norm_master.bias.data
    torch.distributed.broadcast(bias_master, src=0)
    bias = torch.chunk(bias_master, DEPTH)[k]
    norm.bias.data.copy_(bias)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[k]
    A = torch.chunk(A, DEPTH, dim=0)[j]
    A = A.clone()
    A.requires_grad = True

    fwd_start = time.time()
    out = norm(A)
    torch.cuda.synchronize()
    fwd_end = time.time()
    print_rank_0(
        'layer norm forward: pass | {0} --> {1} | {2:.3f} s'.format(tuple(A.shape), tuple(out.shape),
                                                                    fwd_end - fwd_start), logger)

    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = norm_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[k]
    C = torch.chunk(C, DEPTH, dim=0)[j]
    logger.info('Rank {} layernorm forward: {}'.format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[k]
    grad = torch.chunk(grad, DEPTH, dim=0)[j]

    bwd_start = time.time()
    out.backward(grad)
    torch.cuda.synchronize()
    bwd_end = time.time()
    print_rank_0('layer norm backward: pass | {:.3f} s'.format(bwd_end - bwd_start), logger)

    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[k]
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[j]
    logger.info('Rank {} layernorm backward (input_grad): {}'.format(rank, check_equal(A_grad, A.grad)))

    bias_grad = norm_master.weight.grad
    bias_grad = torch.chunk(bias_grad, DEPTH)[k]
    logger.info('Rank {} layernorm backward (weight_grad): {}'.format(rank, check_equal(bias_grad, norm.weight.grad)))

    bias_grad = norm_master.bias.grad
    bias_grad = torch.chunk(bias_grad, DEPTH)[k]
    logger.info('Rank {} layernorm backward (bias_grad): {}'.format(rank, check_equal(bias_grad, norm.bias.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_classifier():
    rank = torch.distributed.get_rank()
    logger = get_dist_logger()
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE

    input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
    weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
    output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)

    j = A_rank = global_context.get_local_rank(input_parallel_mode)
    i = B_rank = global_context.get_local_rank(weight_parallel_mode)
    k = C_rank = global_context.get_local_rank(output_parallel_mode)

    layer = Classifier3D(INPUT_SIZE, NUM_CLASSES, dtype=dtype, bias=True)
    layer = layer.to(device)

    layer_master = VanillaClassifier(INPUT_SIZE, NUM_CLASSES, bias=True, dtype=dtype)
    layer_master = layer_master.to(device)

    weight_master = layer_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=-1)[k]
    layer.weight.data.copy_(weight)
    bias_master = layer_master.bias.data
    torch.distributed.broadcast(bias_master, src=0)
    layer.bias.data.copy_(bias_master)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[k]
    A = torch.chunk(A, DEPTH, dim=0)[j]
    A = A.clone()
    A.requires_grad = True

    fwd_start = time.time()
    out = layer(A)
    torch.cuda.synchronize()
    fwd_end = time.time()
    print_rank_0(
        'head forward: pass | {0} --> {1} | {2:.3f} s'.format(tuple(A.shape), tuple(out.shape), fwd_end - fwd_start),
        logger)
    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = layer_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=0)[j]
    logger.info('Rank {} head forward: {}'.format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=0)[j]
    grad = grad.clone()

    bwd_start = time.time()
    out.backward(grad)
    torch.cuda.synchronize()
    bwd_end = time.time()
    print_rank_0('head backward: pass | {:.3f} s'.format(bwd_end - bwd_start), logger)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[k]
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[j]
    logger.info('Rank {} head backward (input_grad): {}'.format(rank, check_equal(A_grad, A.grad)))

    B_grad = layer_master.weight.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[k]
    if j == k:
        logger.info('Rank {} head backward (weight_grad): {}'.format(rank,
                                                                     check_equal(B_grad, layer.weight.grad)))
    else:
        logger.info('Rank {} head backward (weight_grad): {}'.format(rank, layer.weight.grad is None))

    bias_grad = layer_master.bias.grad
    logger.info('Rank {} head backward (bias_grad): {}'.format(rank, check_equal(bias_grad, layer.bias.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_embed():
    rank = torch.distributed.get_rank()
    device = get_current_device()
    logger = get_dist_logger()
    dtype = torch.float32

    input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
    weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
    output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)

    j = A_rank = global_context.get_local_rank(input_parallel_mode)
    i = B_rank = global_context.get_local_rank(weight_parallel_mode)
    k = C_rank = global_context.get_local_rank(output_parallel_mode)

    layer = PatchEmbedding3D(IMG_SIZE, 4, 3, HIDDEN_SIZE, dtype=dtype)
    torch.nn.init.ones_(layer.cls_token)
    torch.nn.init.ones_(layer.pos_embed)
    layer = layer.to(device)

    layer_master = VanillaPatchEmbedding(IMG_SIZE, 4, 3, HIDDEN_SIZE, dtype=dtype)
    torch.nn.init.ones_(layer_master.cls_token)
    torch.nn.init.ones_(layer_master.pos_embed)
    layer_master = layer_master.to(device)

    proj_weight_master = layer_master.weight.data
    torch.distributed.broadcast(proj_weight_master, src=0)
    proj_weight = torch.chunk(proj_weight_master, DEPTH, dim=0)[k]
    layer.weight.data.copy_(proj_weight)
    proj_bias_master = layer_master.bias.data
    torch.distributed.broadcast(proj_bias_master, src=0)
    proj_bias = torch.chunk(proj_bias_master, DEPTH)[k]
    layer.bias.data.copy_(proj_bias)

    A_shape = (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    A.requires_grad = True

    fwd_start = time.time()
    out = layer(A)
    torch.cuda.synchronize()
    fwd_end = time.time()
    print_rank_0(
        'embedding forward: pass | {0} --> {1} | {2:.3f} s'.format(tuple(A.shape), tuple(out.shape),
                                                                   fwd_end - fwd_start), logger)

    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = layer_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[k]
    C = torch.chunk(C, DEPTH, dim=0)[j]
    logger.info('Rank {} embed forward: {}'.format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[k]
    grad = torch.chunk(grad, DEPTH, dim=0)[j]
    grad = grad.clone()
    bwd_start = time.time()
    out.backward(grad)
    torch.cuda.synchronize()
    bwd_end = time.time()
    print_rank_0('embedding backward: pass | {:.3f} s'.format(bwd_end - bwd_start), logger)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    cls_grad_master = layer_master.cls_token.grad
    cls_grad = torch.chunk(cls_grad_master, DEPTH, dim=-1)[k]
    logger.info('Rank {} embed backward (cls_grad): {}'.format(rank, check_equal(cls_grad, layer.cls_token.grad)))

    pos_grad_master = layer_master.pos_embed.grad
    pos_grad = torch.chunk(pos_grad_master, DEPTH, dim=-1)[k]
    logger.info('Rank {} embed backward (pos_embed_grad): {}'.format(rank, check_equal(pos_grad, layer.pos_embed.grad)))

    B_grad = layer_master.weight.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[k]
    if j == k:
        logger.info('Rank {} embed backward (proj_weight_grad): {}'.format(rank, check_equal(B_grad,
                                                                                             layer.weight.grad)))
    else:
        logger.info('Rank {} embed backward (proj_weight_grad): {}'.format(rank, layer.weight.grad is None))

    bias_grad = layer_master.bias.grad
    bias_grad = torch.chunk(bias_grad, DEPTH)[k]
    logger.info('Rank {} embed backward (proj_bias_grad): {}'.format(rank, check_equal(bias_grad, layer.bias.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_loss():
    rank = torch.distributed.get_rank()
    logger = get_dist_logger()
    device = get_current_device()
    dtype = torch.float32

    input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
    weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
    output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)

    j = A_rank = global_context.get_local_rank(input_parallel_mode)
    i = B_rank = global_context.get_local_rank(weight_parallel_mode)
    k = C_rank = global_context.get_local_rank(output_parallel_mode)

    criterion = CrossEntropyLoss3D()
    criterion_master = torch.nn.CrossEntropyLoss()

    out_shape = (BATCH_SIZE, NUM_CLASSES)
    out_master = torch.randn(out_shape, dtype=dtype, device=device)
    target_master = torch.randint(NUM_CLASSES, (BATCH_SIZE, ), dtype=torch.long, device=device)
    torch.distributed.broadcast(out_master, src=0)
    torch.distributed.broadcast(target_master, src=0)
    out = torch.chunk(out_master, DEPTH, dim=0)[i]
    out = torch.chunk(out, DEPTH, dim=0)[j]
    out = out.clone()
    out.requires_grad = True

    fwd_start = time.time()
    loss = criterion(out, target_master)
    fwd_end = time.time()
    print_rank_0(
        'loss forward: pass | {0} --> {1} | {2:.3f} s'.format(tuple(out.shape), tuple(loss.shape), fwd_end - fwd_start),
        logger)

    out_master = out_master.clone()
    out_master.requires_grad = True
    loss_master = criterion_master(out_master, target_master)
    logger.info('Rank {} CrossEntropyLoss forward: {}'.format(rank, check_equal(loss, loss_master)))

    bwd_start = time.time()
    loss.backward()
    bwd_end = time.time()
    print_rank_0('loss backward: pass | {:.3f} s'.format(bwd_end - bwd_start), logger)

    loss_master.backward()
    out_grad = out_master.grad
    out_grad = torch.chunk(out_grad, DEPTH, dim=0)[i]
    out_grad = torch.chunk(out_grad, DEPTH, dim=0)[j]
    logger.info('Rank {} CrossEntropyLoss backward: {}'.format(rank, check_equal(out_grad, out.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start
