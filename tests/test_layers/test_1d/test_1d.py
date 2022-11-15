#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
from checks_1d.check_layer_1d import *

from colossalai.core import global_context as gpc
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port

CONFIG = dict(parallel=dict(pipeline=dict(size=1), tensor=dict(size=4, mode='1d')),)


def check_layer(rank, world_size, port):
    disable_existing_loggers()
    launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    check_linear_col()
    check_linear_row()
    check_embed()
    check_vocab_parallel_embed()
    check_classifier_no_given_weight()
    check_vocab_parallel_classifier_no_given_weight()
    check_classifier_given_embed_weight()
    check_vocab_parallel_classifier_given_embed_weight()
    check_vocab_parallel_loss()

    check_linear_row_stream_inference()

    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_1d():
    world_size = 4
    run_func = partial(check_layer, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_1d()
