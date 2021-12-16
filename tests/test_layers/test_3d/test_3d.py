#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.initialize import launch, get_default_parser

from checks_3d.check_layer_3d import *
from checks_3d.check_operation_3d import *
from colossalai.logging import get_dist_logger
from functools import partial

CONFIG = dict(parallel=dict(pipeline=1, tensor=dict(mode='3d', size=8)),
              seed=0)


# def check_operations():
#     check_AB()
#     check_ABT()
#     check_ATB()
#     check_add()
#     check_mul()
#     check_sum()


def check_layer():
    logger = get_dist_logger()
    liear_fwd_time, linear_bwd_time = check_linear()
    norm_fwd_time, norm_bwd_time = check_layernorm()
    attn_fwd_time, attn_bwd_time = check_attention()
    mlp_fwd_time, mlp_bwd_time = check_mlp()
    head_fwd_time, head_bwd_time = check_head()
    embed_fwd_time, embed_bwd_time = check_embed()
    loss_fwd_time, loss_bwd_time = check_loss()
    block_fwd_time = norm_fwd_time + attn_fwd_time + norm_fwd_time + mlp_fwd_time
    block_bwd_time = norm_bwd_time + attn_bwd_time + norm_bwd_time + mlp_bwd_time
    fwd_time = embed_fwd_time + NUM_BLOCKS * block_fwd_time + norm_fwd_time + head_fwd_time + loss_fwd_time
    bwd_time = embed_bwd_time + NUM_BLOCKS * block_bwd_time + norm_bwd_time + head_bwd_time + loss_bwd_time
    logger.info('ViT forward time: {:.3f} s | backward time: {:.3f} s'.format(
        fwd_time, bwd_time),
        ranks=[0])


def check_layer_and_operation(rank, world_size):
    launch(config=CONFIG,
           rank=rank,
           world_size=world_size,
           host='localhost',
           port=29923,
           backend='nccl')

    check_layer()
    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_3d():
    world_size = 8
    run_func = partial(check_layer_and_operation, world_size=world_size)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_3d()
