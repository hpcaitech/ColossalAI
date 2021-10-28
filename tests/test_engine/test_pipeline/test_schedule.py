#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os.path as osp

import pytest

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
    model, train_dataloader, test_dataloader, criterion, optimizer, schedule, lr_scheduler = initialize(CONFIG_PATH)
    logger = get_global_dist_logger()

    schedule.zero_grad()
    output, label, losses = schedule.forward_backward_step(forward_only=False)
    schedule.step()
    logger.info('losses: {}'.format([loss.item() for loss in losses]))

    gpc.destroy()
    logger.info('training finished')


if __name__ == '__main__':
    test_schedule()
