#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os.path as osp

import pytest
import torch

from colossalai import initialize
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_global_dist_logger

NUM_BATCH = 128

BATCH_SIZE = 32
SEQ_LENGTH = 128
HIDDEN_SIZE = 512

DIR_PATH = osp.dirname(osp.realpath(__file__))
PIPE_CONFIG_PATH = osp.join(DIR_PATH, '../configs/pipeline_vanilla_resnet.py')


def run_pipeline(config):
    engine, train_dataloader, test_dataloader = initialize(config)
    logger = get_global_dist_logger()
    rank = torch.distributed.get_rank()

    engine.train()
    outputs, labels, loss = engine.step(iter(train_dataloader))
    if gpc.is_last_rank(ParallelMode.PIPELINE):
        logger.info('losses: {}'.format(rank, loss.item()))

    gpc.destroy()
    logger.info('Test engine pipeline finished')


@pytest.mark.skip("This test should be invoked using the test.sh provided")
@pytest.mark.dist
def test_engine():
    run_pipeline(PIPE_CONFIG_PATH)


if __name__ == '__main__':
    test_engine()
