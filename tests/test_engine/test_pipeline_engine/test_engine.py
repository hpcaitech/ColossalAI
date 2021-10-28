#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os.path as osp

import pytest
import torch

from colossalai import initialize
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine import Engine
from colossalai.logging import get_global_dist_logger

NUM_BATCH = 128

BATCH_SIZE = 32
SEQ_LENGTH = 128
HIDDEN_SIZE = 512

DIR_PATH = osp.dirname(osp.realpath(__file__))
PIPE_CONFIG_PATH = osp.join(DIR_PATH, '../configs/pipeline_vanilla_resnet.py')


def run_pipeline(config):
    model, train_dataloader, test_dataloader, criterion, optimizer, schedule, lr_scheduler = initialize(config)
    logger = get_global_dist_logger()
    rank = torch.distributed.get_rank()
    engine = Engine(model=model,
                    train_dataloader=train_dataloader,
                    criterion=criterion,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    schedule=schedule)

    engine.train()
    logger.info('lr = %g' % engine.get_lr())
    outputs, labels, loss = engine.step()
    if gpc.is_last_rank(ParallelMode.PIPELINE):
        logger.info('losses: {}'.format(rank, loss.item()))
    logger.info('lr = %g' % engine.get_lr())

    gpc.destroy()
    logger.info('Test engine pipeline finished')


@pytest.mark.skip("This test should be invoked using the test.sh provided")
@pytest.mark.dist
def test_engine():
    run_pipeline(PIPE_CONFIG_PATH)


if __name__ == '__main__':
    test_engine()
