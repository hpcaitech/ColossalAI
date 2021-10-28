#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os.path as osp

import pytest
import torch

from colossalai import initialize
from colossalai.core import global_context as gpc
from colossalai.engine import Engine
from colossalai.logging import get_global_dist_logger
from colossalai.utils import report_memory_usage

NUM_BATCH = 128
NUM_MICRO = 6

BATCH_SIZE = 32
SEQ_LENGTH = 128
HIDDEN_SIZE = 512

DIR_PATH = osp.dirname(osp.realpath(__file__))
NO_PIPE_CONFIG_PATH = osp.join(DIR_PATH, '../configs/non_pipeline_resnet_torch_amp.py')


def test_no_pipeline(config):
    print('Test no pipeline engine start')

    model, train_dataloader, test_dataloader, criterion, optimizer, schedule, lr_scheduler = initialize(config)
    logger = get_global_dist_logger()

    rank = torch.distributed.get_rank()
    engine = Engine(model=model,
                    train_dataloader=train_dataloader,
                    criterion=criterion,
                    optimizer=optimizer,
                    schedule=schedule)

    engine.train()
    logger.info('lr = %g' % engine.get_lr())
    output, label, loss = engine.step()
    logger.info('Rank {} returns: {}'.format(rank, loss.item()))
    logger.info('lr = %g' % engine.get_lr())

    gpc.destroy()
    logger.info('Test engine finished')
    report_memory_usage("After testing")


@pytest.mark.skip("This test should be invoked using the test.sh provided")
@pytest.mark.dist
def test_engine():
    test_no_pipeline(NO_PIPE_CONFIG_PATH)


if __name__ == '__main__':
    test_engine()
