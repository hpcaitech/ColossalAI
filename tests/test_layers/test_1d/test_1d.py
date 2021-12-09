#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest

from colossalai.core import global_context as gpc
from colossalai.initialize import launch, get_default_parser
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
    parser = get_default_parser()
    args = parser.parse_args()
    launch(config=CONFIG,
           rank=args.rank,
           world_size=args.world_size,
           host=args.host,
           port=args.port,
           backend=args.backend)

    check_layer()
    gpc.destroy()


if __name__ == '__main__':
    test_1d()
