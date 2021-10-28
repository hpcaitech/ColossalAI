#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import time

import numpy as np
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context
from colossalai.logging import get_global_dist_logger
from colossalai.registry import LAYERS, LOSSES
from colossalai.utils import get_current_device, print_rank_0

from common import *


def check_linear():
    rank = torch.distributed.get_rank()
    logger = get_global_dist_logger()
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = 2 * HIDDEN_SIZE

    j = A_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
    i = B_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
    k = C_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)

    layer = LAYERS.get_module('Linear3D')(INPUT_SIZE,
                                          OUTPUT_SIZE,
                                          ParallelMode.PARALLEL_3D_INPUT,
                                          ParallelMode.PARALLEL_3D_WEIGHT,
                                          dtype=dtype,
                                          bias=True)
    torch.nn.init.zeros_(layer.bias)
    torch.nn.init.ones_(layer.weight)
    layer = layer.to(device)
    layer_master = torch.nn.Linear(INPUT_SIZE, OUTPUT_SIZE)
    torch.nn.init.zeros_(layer_master.bias)
    torch.nn.init.ones_(layer_master.weight)
    layer_master = layer_master.to(device)

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
    fwd_end = time.time()
    print_rank_0(
        'linear forward: {0} --> {1} | {2:.3f} s'.format(
            tuple(A.shape), tuple(out.shape), fwd_end - fwd_start), logger)
    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = layer_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]
    C = torch.chunk(C, DEPTH, dim=0)[k]
    logger.info('Rank {} linear forward: {}'.format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape,
                              dtype=dtype,
                              device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    grad = torch.chunk(grad, DEPTH, dim=0)[k]

    bwd_start = time.time()
    out.backward(grad)
    bwd_end = time.time()
    print_rank_0('linear backward: {:.3f} s'.format(bwd_end - bwd_start),
                 logger)

    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[k]
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[j]
    logger.info('Rank {} linear backward (input_grad): {}'.format(
        rank, check_equal(A_grad, A.grad)))

    B_grad = layer_master.weight.grad.transpose(0, 1)
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[k]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[j]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[i]
    logger.info('Rank {} linear backward (weight_grad): {}'.format(
        rank, check_equal(B_grad, layer.weight.grad)))

    if j == k:
        bias_grad = layer_master.bias.grad
        bias_grad = torch.chunk(bias_grad, DEPTH)[j]
        bias_grad = torch.chunk(bias_grad, DEPTH)[i]
        logger.info('Rank {} linear backward (bias_grad): {}'.format(
            rank, check_equal(bias_grad, layer.bias.grad)))
    else:
        logger.info('Rank {} linear backward (bias_grad): {}'.format(
            rank,
            # np.count_nonzero(layer.bias.grad.detach().cpu().numpy()) == 0))
            layer.bias.grad is None))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_layernorm():
    rank = torch.distributed.get_rank()
    logger = get_global_dist_logger()
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE

    j = A_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
    i = B_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
    k = C_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)

    norm = LAYERS.get_module('LayerNorm3D')(INPUT_SIZE,
                                            ParallelMode.PARALLEL_3D_INPUT,
                                            ParallelMode.PARALLEL_3D_WEIGHT,
                                            eps=1e-6,
                                            dtype=dtype)
    norm = norm.to(device)
    norm_master = torch.nn.LayerNorm(INPUT_SIZE, eps=1e-6)
    norm_master = norm_master.to(device)

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
    fwd_end = time.time()
    print_rank_0(
        'layer norm forward: pass | {0} --> {1} | {2:.3f} s'.format(
            tuple(A.shape), tuple(out.shape), fwd_end - fwd_start), logger)

    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = norm_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[k]
    C = torch.chunk(C, DEPTH, dim=0)[j]
    logger.info('Rank {} layernorm forward: {}'.format(rank,
                                                       check_equal(out, C)))
    # time.sleep(rank)
    # logger.info('Rank {0} master:\n{1}\nRank {0} out:\n{2}\nRank {0} true:\n{3}\n'.
    #       format(rank,
    #              C_master.detach().cpu().numpy().tolist(),
    #              out.detach().cpu().numpy().tolist(),
    #              C.detach().cpu().numpy().tolist()))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[k]
    grad = torch.chunk(grad, DEPTH, dim=0)[j]

    bwd_start = time.time()
    out.backward(grad)
    bwd_end = time.time()
    print_rank_0(
        'layer norm backward: pass | {:.3f} s'.format(bwd_end - bwd_start),
        logger)

    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[k]
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[j]
    logger.info('Rank {} layernorm backward (input_grad): {}'.format(
        rank, check_equal(A_grad, A.grad)))

    if j == k:
        bias_grad = norm_master.weight.grad
        bias_grad = torch.chunk(bias_grad, DEPTH)[j]
        bias_grad = torch.chunk(bias_grad, DEPTH)[i]
        logger.info('Rank {} linear backward (weight_grad): {}'.format(
            rank, check_equal(bias_grad, norm.weight.grad)))
    else:
        logger.info('Rank {} linear backward (weight_grad): {}'.format(
            rank,
            # np.count_nonzero(layer.bias.grad.detach().cpu().numpy()) == 0))
            norm.weight.grad is None))

    if j == k:
        bias_grad = norm_master.bias.grad
        bias_grad = torch.chunk(bias_grad, DEPTH)[j]
        bias_grad = torch.chunk(bias_grad, DEPTH)[i]
        logger.info('Rank {} linear backward (bias_grad): {}'.format(
            rank, check_equal(bias_grad, norm.bias.grad)))
    else:
        logger.info('Rank {} linear backward (bias_grad): {}'.format(
            rank,
            # np.count_nonzero(layer.bias.grad.detach().cpu().numpy()) == 0))
            norm.bias.grad is None))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_attention():
    rank = torch.distributed.get_rank()
    device = get_current_device()
    logger = get_global_dist_logger()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    NUM_ATTENTION_HEADS = 2

    j = A_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
    i = B_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
    k = C_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)

    layer = LAYERS.get_module('ViTSelfAttention3D')(HIDDEN_SIZE,
                                                    NUM_ATTENTION_HEADS,
                                                    0.,
                                                    0.1,
                                                    dtype=dtype,
                                                    bias=True)
    layer = layer.to(device)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[k]
    A = torch.chunk(A, DEPTH, dim=0)[j]
    A = A.clone()
    A.requires_grad = True

    mask_shape = (BATCH_SIZE // DEPTH, NUM_ATTENTION_HEADS // DEPTH,
                  SEQ_LENGTH // DEPTH, SEQ_LENGTH // DEPTH)
    attention_mask = torch.zeros(mask_shape, dtype=dtype, device=device)

    fwd_start = time.time()
    out = layer(A)
    fwd_end = time.time()
    print_rank_0(
        'self attention forward: pass | {0} --> {1} | {2:.3f} s'.format(
            tuple(A.shape), tuple(out.shape), fwd_end - fwd_start), logger)

    grad_shape = out.shape
    grad = torch.randn(grad_shape, dtype=dtype, device=device)

    bwd_start = time.time()
    out.backward(grad)
    bwd_end = time.time()
    print_rank_0(
        'self attention backward: pass | {:.3f} s'.format(bwd_end - bwd_start),
        logger)

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_mlp():
    rank = torch.distributed.get_rank()
    device = get_current_device()
    logger = get_global_dist_logger()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE

    j = A_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
    i = B_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
    k = C_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)

    layer = LAYERS.get_module('ViTMLP3D')(HIDDEN_SIZE,
                                          1,
                                          0.1,
                                          'gelu',
                                          dtype=dtype,
                                          bias=True)

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
    fwd_end = time.time()
    print_rank_0(
        'mlp forward: pass | {0} --> {1} | {2:.3f} s'.format(
            tuple(A.shape), tuple(out.shape), fwd_end - fwd_start), logger)

    grad_shape = out.shape
    grad = torch.randn(grad_shape, dtype=dtype, device=device)

    bwd_start = time.time()
    out.backward(grad)
    bwd_end = time.time()
    print_rank_0('mlp backward: pass | {:.3f} s'.format(bwd_end - bwd_start),
                 logger)

    return fwd_end - fwd_start, bwd_end - bwd_start


class Testvithead(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = x[:, 0]
        x = self.linear(x)
        return x


def check_head():
    rank = torch.distributed.get_rank()
    logger = get_global_dist_logger()
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE

    j = A_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
    i = B_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
    k = C_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)

    head = LAYERS.get_module('ViTHead3D')(INPUT_SIZE,
                                          NUM_CLASSES,
                                          dtype=dtype,
                                          bias=True)
    torch.nn.init.zeros_(head.linear.bias)
    torch.nn.init.ones_(head.linear.weight)
    head = head.to(device)

    layer = Testvithead(INPUT_SIZE, NUM_CLASSES, bias=True)
    torch.nn.init.zeros_(layer.linear.bias)
    torch.nn.init.ones_(layer.linear.weight)
    layer = layer.to(device)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[k]
    A = torch.chunk(A, DEPTH, dim=0)[j]
    A = A.clone()
    A.requires_grad = True

    fwd_start = time.time()
    out = head(A)
    fwd_end = time.time()
    print_rank_0(
        'head forward: pass | {0} --> {1} | {2:.3f} s'.format(
            tuple(A.shape), tuple(out.shape), fwd_end - fwd_start), logger)
    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = layer(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]
    C = torch.chunk(C, DEPTH, dim=0)[k]
    logger.info('Rank {} head forward: {}'.format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape,
                              dtype=dtype,
                              device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    grad = torch.chunk(grad, DEPTH, dim=0)[k]

    bwd_start = time.time()
    out.backward(grad)
    bwd_end = time.time()
    print_rank_0('head backward: pass | {:.3f} s'.format(bwd_end - bwd_start),
                 logger)

    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[k]
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[j]
    # if j == 0:
    logger.info('Rank {} head backward (input_grad): {}'.format(
        rank, check_equal(A_grad, A.grad)))
    # else:
    #     logger.info('Rank {} head backward (input_grad): {}'.format(
    #         # rank, check_equal(A_grad, A.grad)))
    #         rank,
    #         A.grad is None))

    B_grad = layer.linear.weight.grad.transpose(0, 1)
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[k]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[j]
    pad_shape = (B_grad.shape[0], math.ceil(B_grad.shape[-1] / DEPTH) * DEPTH -
                 B_grad.shape[-1])
    B_grad = torch.cat(
        [B_grad, torch.zeros(pad_shape, dtype=dtype, device=device)], dim=-1)
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[i]
    logger.info('Rank {} head backward (weight_grad): {}'.format(
        rank, check_equal(B_grad, head.linear.weight.grad)))

    if j == k:
        bias_grad = layer.linear.bias.grad
        bias_grad = torch.chunk(bias_grad, DEPTH)[j]
        pad_shape = (math.ceil(bias_grad.shape[0] / DEPTH) * DEPTH -
                     bias_grad.shape[0], )
        bias_grad = torch.cat(
            [bias_grad,
             torch.zeros(pad_shape, dtype=dtype, device=device)])
        bias_grad = torch.chunk(bias_grad, DEPTH)[i]
        logger.info('Rank {} head backward (bias_grad): {}'.format(
            rank, check_equal(bias_grad, head.linear.bias.grad)))
    else:
        logger.info('Rank {} head backward (bias_grad): {}'.format(
            rank,
            # np.count_nonzero(
            #     head.linear.bias.grad.detach().cpu().numpy()) == 0))
            head.linear.bias.grad is None))

    return fwd_end - fwd_start, bwd_end - bwd_start


class Testvitembed(torch.nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int,
                 embed_size: int, drop_prob: float) -> None:
        super().__init__()
        self.proj = torch.nn.Conv2d(in_chans,
                                    embed_size,
                                    kernel_size=patch_size,
                                    stride=patch_size)
        num_patches = (img_size // patch_size)**2
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_size))
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_size))
        self.pos_drop = torch.nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x


def check_embed():
    rank = torch.distributed.get_rank()
    device = get_current_device()
    logger = get_global_dist_logger()
    dtype = torch.float32

    j = A_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
    i = B_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
    k = C_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)

    layer = LAYERS.get_module('ViTPatchEmbedding3D')(IMG_SIZE, 4, 3,
                                                     HIDDEN_SIZE, 0.)
    torch.nn.init.zeros_(layer.proj.bias)
    torch.nn.init.ones_(layer.proj.weight)
    torch.nn.init.ones_(layer.cls_token)
    torch.nn.init.ones_(layer.pos_embed)
    layer = layer.to(device)

    layer_master = Testvitembed(IMG_SIZE, 4, 3, HIDDEN_SIZE, 0.)
    torch.nn.init.zeros_(layer_master.proj.bias)
    torch.nn.init.ones_(layer_master.proj.weight)
    torch.nn.init.ones_(layer_master.cls_token)
    torch.nn.init.ones_(layer_master.pos_embed)
    layer_master = layer_master.to(device)

    A_shape = (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    A.requires_grad = True

    fwd_start = time.time()
    out = layer(A)
    fwd_end = time.time()
    print_rank_0(
        'embedding forward: pass | {0} --> {1} | {2:.3f} s'.format(
            tuple(A.shape), tuple(out.shape), fwd_end - fwd_start), logger)
    # out_cls = out[:, 0]
    # out_tensor = out[:, 1:]

    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = layer_master(A_master)
    # if j == 0:
    #     C_cls = C_master[:, 0]
    #     C_cls = torch.chunk(C_cls, DEPTH, dim=0)[i]
    #     C_cls = torch.chunk(C_cls, DEPTH, dim=-1)[k]
    #     logger.info('Rank {} embed forward (cls): {}'.format(
    #         rank, check_equal(out_cls, C_cls)))
    # C = C_master[:, 1:]
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[k]
    C = torch.chunk(C, DEPTH, dim=0)[j]
    logger.info('Rank {} embed forward: {}'.format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape,
                              dtype=dtype,
                              device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    # cls_grad = grad_master[:, 0]
    # cls_grad = torch.chunk(cls_grad, DEPTH, dim=0)[i]
    # cls_grad = torch.chunk(cls_grad, DEPTH, dim=-1)[k]
    # grad = grad_master[:, 1:]
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[k]
    grad = torch.chunk(grad, DEPTH, dim=0)[j]
    # grad = torch.cat((torch.unsqueeze(cls_grad, 1), grad), dim=1)
    bwd_start = time.time()
    out.backward(grad)
    bwd_end = time.time()
    print_rank_0(
        'embedding backward: pass | {:.3f} s'.format(bwd_end - bwd_start),
        logger)

    C_master.backward(grad_master)
    # A_grad = A_master.grad
    # logger.info('Rank {} embed backward (input_grad): {}'.format(
    #     rank, check_equal(A_grad, A.grad)))
    # time.sleep(0.1 * rank)
    # logger.info(
    #     'Rank {0} master:\n{1}\nRank {0} out:\n{2}\nRank {0} true:\n{3}\n'.
    #     format(rank,
    #            A_master.grad.detach().cpu().numpy().tolist(),
    #            A.grad.detach().cpu().numpy().tolist(),
    #            A_grad.detach().cpu().numpy().tolist()), ranks=[0])

    cls_grad_master = layer_master.cls_token.grad
    cls_grad = torch.chunk(cls_grad_master, DEPTH, dim=-1)[k]
    # if j == 0:
    logger.info('Rank {} embed backward (cls_grad): {}'.format(
        rank, check_equal(cls_grad, layer.cls_token.grad)))
    # else:.
    #     logger.info('Rank {} embed backward (cls_grad): {}'.format(
    #         rank,
    #         layer.cls_token.grad is None or np.count_nonzero(
    #             layer.cls_token.grad.detach().cpu().numpy()) == 0))

    pos_grad_master = layer_master.pos_embed.grad
    pos_grad = torch.chunk(pos_grad_master, DEPTH, dim=-1)[k]
    logger.info('Rank {} embed backward (pos_embed_grad): {}'.format(
        rank, check_equal(pos_grad, layer.pos_embed.grad)))
    # if i == 0:
    #     pos_cls_grad = pos_grad[:, 0]
    #     pos_tensor_grad = pos_grad[:, 1:]
    #     pos_tensor_grad = torch.chunk(pos_tensor_grad, DEPTH, dim=1)[j]
    #     if j == 0:
    #         logger.info('Rank {} embed backward (pos_embed_grad): {}'.format(
    #             rank,
    #             check_equal(
    #                 torch.cat(
    #                     (torch.unsqueeze(pos_cls_grad, 1), pos_tensor_grad),
    #                     dim=1), layer.pos_embed.grad)))
    #     else:
    #         logger.info('Rank {} embed backward (pos_embed_grad): {}'.format(
    #             rank, check_equal(pos_tensor_grad, layer.pos_embed.grad[:,
    #                                                                     1:])))
    # else:
    #     logger.info('Rank {} embed backward (pos_embed_grad): {}'.format(
    #         rank, layer.pos_embed.grad is None))

    B_grad = layer_master.proj.weight.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[k]
    logger.info('Rank {} embed backward (proj_weight_grad): {}'.format(
        rank, check_equal(B_grad, layer.proj.weight.grad)))

    bias_grad = layer_master.proj.bias.grad
    bias_grad = torch.chunk(bias_grad, DEPTH)[k]
    logger.info('Rank {} embed backward (proj_bias_grad): {}'.format(
        rank, check_equal(bias_grad, layer.proj.bias.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_loss():
    rank = torch.distributed.get_rank()
    logger = get_global_dist_logger()
    device = get_current_device()
    dtype = torch.float32

    j = A_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)
    i = B_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)
    k = C_rank = global_context.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)

    criterion = LOSSES.get_module('CrossEntropyLoss3D')(
        ParallelMode.PARALLEL_3D_INPUT, ParallelMode.PARALLEL_3D_WEIGHT)
    criterion_master = torch.nn.CrossEntropyLoss()

    out_shape = (BATCH_SIZE, NUM_CLASSES)
    out_master = torch.randn(out_shape, dtype=dtype, device=device)
    target_master = torch.randint(NUM_CLASSES, (BATCH_SIZE, ),
                                  dtype=torch.long,
                                  device=device)
    torch.distributed.broadcast(out_master, src=0)
    torch.distributed.broadcast(target_master, src=0)
    out = torch.chunk(out_master, DEPTH, dim=0)[i]
    out = torch.chunk(out, DEPTH, dim=-1)[k]
    out = torch.chunk(out, DEPTH, dim=0)[j]
    out = out.clone()
    out.requires_grad = True

    fwd_start = time.time()
    loss = criterion(out, target_master)
    fwd_end = time.time()
    print_rank_0(
        'loss forward: pass | {0} --> {1} | {2:.3f} s'.format(
            tuple(out.shape), tuple(loss.shape), fwd_end - fwd_start), logger)

    out_master = out_master.clone()
    out_master.requires_grad = True
    loss_master = criterion_master(out_master, target_master)
    logger.info('Rank {} CrossEntropyLoss forward: {}'.format(
        rank, check_equal(loss, loss_master)))

    bwd_start = time.time()
    loss.backward()
    bwd_end = time.time()
    print_rank_0('loss backward: pass | {:.3f} s'.format(bwd_end - bwd_start),
                 logger)

    loss_master.backward()
    out_grad = out_master.grad
    out_grad = torch.chunk(out_grad, DEPTH, dim=0)[i]
    out_grad = torch.chunk(out_grad, DEPTH, dim=-1)[k]
    out_grad = torch.chunk(out_grad, DEPTH, dim=0)[j]
    logger.info('Rank {} CrossEntropyLoss backward: {}'.format(
        rank, check_equal(out_grad, out.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start
