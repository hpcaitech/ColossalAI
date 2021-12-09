#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from colossalai.initialize import launch, get_default_parser
from colossalai.logging import get_dist_logger
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
    parser = get_default_parser()
    args = parser.parse_args()
    launch(config=CONFIG,
           rank=args.rank,
           world_size=args.world_size,
           host=args.host,
           port=args.port,
           backend=args.backend)
    logger = get_dist_logger()
    logger.info('Distributed environment is initialzied.', ranks=[0])

    torch.backends.cudnn.benchmark = True

    # check layers
    check_layer()


if __name__ == '__main__':
    _test_main()
