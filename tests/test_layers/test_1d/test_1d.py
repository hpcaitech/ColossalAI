#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest

from colossalai.core import global_context as gpc
from colossalai.initialize import init_dist
from test_layer import *

CONFIG = dict(
    parallel=dict(
        pipeline=dict(size=1),
        tensor=dict(
            size=2,
            mode='1d'
        )
    ),
)


def check_layer():
    # print_rank_0('start check_linear_col')
    check_linear_col()
    check_linear_row()
    check_attention()
    check_mlp()
    check_patch_embedding()
    check_embed()
    check_head()

@pytest.mark.dist
@pytest.mark.skip("This test should be invoked by test.sh in the same folder as it runs on multiple gpus")
def test_1d():
    init_dist(config=CONFIG)
    gpc.set_seed()
    
    check_layer()
    gpc.destroy()


if __name__ == '__main__':
    test_1d()
