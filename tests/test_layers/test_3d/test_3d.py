#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
from colossalai.core import global_context as gpc
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.utils import free_port
from colossalai.testing import rerun_if_address_is_in_use
from checks_3d.check_layer_3d import (check_classifier_given_embed_weight, check_classifier_no_given_weight,
                                      check_embed, check_layernorm, check_linear, check_loss, check_patch_embed,
                                      check_vocab_parallel_classifier_given_embed_weight,
                                      check_vocab_parallel_classifier_no_given_weight, check_vocab_parallel_embed,
                                      check_vocab_parallel_loss)

CONFIG = dict(
    parallel=dict(
        pipeline=1,
        tensor=dict(mode='3d', size=8),
    ),
    seed=42,
)


def check_layer():
    check_linear()
    check_layernorm()
    check_classifier_no_given_weight()
    check_vocab_parallel_classifier_no_given_weight()
    check_classifier_given_embed_weight()
    check_vocab_parallel_classifier_given_embed_weight()
    check_embed()
    check_patch_embed()
    check_vocab_parallel_embed()
    check_loss()
    check_vocab_parallel_loss()


def check_layer_and_operation(rank, world_size, port):
    disable_existing_loggers()
    launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    check_layer()
    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_3d():
    world_size = 8
    run_func = partial(check_layer_and_operation, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_3d()
