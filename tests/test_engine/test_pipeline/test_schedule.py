#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os.path as osp

import pytest

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.initialize import initialize
from colossalai.logging import get_global_dist_logger

NUM_BATCH = 128

BATCH_SIZE = 32
SEQ_LENGTH = 128
HIDDEN_SIZE = 512

DIR_PATH = osp.dirname(osp.realpath(__file__))
CONFIG_PATH = osp.join(DIR_PATH, '../configs/pipeline_vanilla_resnet.py')


@pytest.mark.skip("This test should be invoked using the test.sh provided")
@pytest.mark.dist
def test_schedule():
    engine, train_dataloader, test_dataloader = initialize(CONFIG_PATH)
    logger = get_global_dist_logger()

    model = engine.model
    optimizer = engine.optimizer
    criterion = engine.criterion
    schedule = engine._schedule

    output, label, loss = schedule.forward_backward_step(
        data_iter=iter(train_dataloader),
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        forward_only=False
    )
    schedule.optimizer_step(model, optimizer)

    if gpc.is_last_rank(ParallelMode.PIPELINE):
        logger.info('losses: {}'.format(loss))

    gpc.destroy()
    logger.info('training finished')


if __name__ == '__main__':
    test_schedule()
