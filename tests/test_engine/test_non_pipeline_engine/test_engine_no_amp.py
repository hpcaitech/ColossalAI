#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os.path as osp

import pytest
import torch

from colossalai import initialize
from colossalai.core import global_context as gpc
from colossalai.logging import get_global_dist_logger
from colossalai.utils import report_memory_usage

NUM_BATCH = 128
NUM_MICRO = 6

BATCH_SIZE = 32
SEQ_LENGTH = 128
HIDDEN_SIZE = 512

DIR_PATH = osp.dirname(osp.realpath(__file__))
NO_PIPE_CONFIG_PATH = osp.join(DIR_PATH, '../configs/non_pipeline_resnet.py')


def test_no_pipeline(config):
    print('Test no pipeline engine start')

    engine, train_dataloader, test_dataloader = initialize(config)
    logger = get_global_dist_logger()

    rank = torch.distributed.get_rank()

    engine.train()
    output, label, loss = engine.step(iter(train_dataloader))
    logger.info('Rank {} returns: {}'.format(rank, loss.item()))

    gpc.destroy()
    logger.info('Test engine finished')
    report_memory_usage("After testing")


@pytest.mark.skip("This test should be invoked using the test.sh provided")
@pytest.mark.dist
def test_engine():
    test_no_pipeline(NO_PIPE_CONFIG_PATH)


if __name__ == '__main__':
    test_engine()
