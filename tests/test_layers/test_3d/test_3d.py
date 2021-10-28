#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from colossalai.initialize import init_dist

from test_layer import *
from test_operation import *

CONFIG = dict(parallel=dict(pipeline=1, tensor=dict(mode='3d', size=8)),
              seed=0)


def check_operations():
    check_AB()
    check_ABT()
    check_ATB()
    check_add()
    check_mul()
    check_sum()
    # check_pooler()


def check_layer():
    logger = get_global_dist_logger()
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


def _test_main():
    # init dist
    init_dist(CONFIG)
    logger = get_global_dist_logger()
    logger.info('Distributed environment is initialzied.', ranks=[0])

    global_context.set_seed()
    torch.backends.cudnn.benchmark = True

    # check operation
    check_operations()

    # check layers
    check_layer()


if __name__ == '__main__':
    _test_main()
