#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from pathlib import Path

import pytest
import torch.autograd

import colossalai
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine import Engine
from colossalai.logging import get_global_dist_logger
from colossalai.nn.layer._parallel_utilities import _gather

level = os.environ['LEVEL']
CONFIG_PATH = Path(__file__).parent.parent.joinpath(f'configs/vit_2d_zero{level}.py')


def eval_epoch(engine: Engine, test_dataloader):
    engine.eval()
    accumulated_loss = 0
    correct_sum = 0
    total_sum = 0
    num_steps = len(test_dataloader)
    data_iter = iter(test_dataloader)

    for i in range(num_steps):
        output, label, loss = engine.step(data_iter)
        accumulated_loss += loss.detach().cpu().numpy()

        output = _gather(
            output[0],
            ParallelMode.PARALLEL_2D_ROW,
            1
        )
        output = _gather(
            output,
            ParallelMode.PARALLEL_2D_COL,
            0,
        )
        output = torch.argmax(output, dim=-1)
        correct = torch.sum(label[0] == output)
        correct_sum += correct
        total_sum += label[0].size(0)
    avg_loss = accumulated_loss / num_steps
    return correct_sum, total_sum, avg_loss


def train_epoch(engine, train_dataloader):
    engine.train()
    accumulated_loss = 0
    num_steps = len(train_dataloader)
    data_iter = iter(train_dataloader)
    for i in range(num_steps):
        output, label, loss = engine.step(data_iter)
        accumulated_loss += loss.detach().cpu().numpy()
    avg_loss = accumulated_loss / num_steps
    return avg_loss


@pytest.mark.dist
@pytest.mark.skip("This test should be invoked by test.sh in the same folder as it runs on multiple gpus")
def test_2d_parallel_vision_transformer():
    # init dist
    engine, train_dataloader, test_dataloader = colossalai.initialize(CONFIG_PATH)
    logger = get_global_dist_logger()

    logger.info('start training')
    for epoch in range(gpc.config.num_epochs):
        train_loss = train_epoch(engine, train_dataloader)

        logger.info(f'epoch {epoch} - train loss: {train_loss}')

        if epoch % 2 == 0:
            correct_sum, total_sum, eval_loss = eval_epoch(engine, test_dataloader)
            logger.info(
                f'epoch {epoch} - eval loss: {eval_loss}, total: {total_sum}, '
                f'correct: {correct_sum}, acc: {correct_sum / total_sum}')


if __name__ == '__main__':
    test_2d_parallel_vision_transformer()
