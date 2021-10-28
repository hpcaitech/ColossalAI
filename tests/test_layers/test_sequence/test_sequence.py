#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from colossalai.initialize import init_dist
from colossalai.logging import get_global_dist_logger
from test_layer import *

CONFIG = dict(
    parallel=dict(
        pipeline=1,
        tensor=dict(mode='sequence', size=4)
    )
)


def check_layer():
    check_selfattention()


def _test_main():
    # init dist
    init_dist(CONFIG)
    logger = get_global_dist_logger()
    logger.info('Distributed environment is initialzied.', ranks=[0])

    gpc.set_seed()
    torch.backends.cudnn.benchmark = True

    # check layers
    check_layer()


if __name__ == '__main__':
    _test_main()
