#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from pathlib import Path

import pytest
import torch.autograd

import colossalai
from colossalai.builder import build_lr_scheduler
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_global_dist_logger
from colossalai.nn.layer._parallel_utilities import _gather

CONFIG_PATH = Path(__file__).parent.parent.joinpath('configs/vit_2d.py')


def eval(engine, test_dataloader):
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


def train(engine, train_dataloader, lr_scheduler):
    engine.train()
    accumulated_loss = 0
    num_steps = len(train_dataloader)
    data_iter = iter(train_dataloader)

    for i in range(num_steps):
        output, label, loss = engine.step(data_iter)
        accumulated_loss += loss.squeeze(0).detach().cpu().numpy()
    avg_loss = accumulated_loss / num_steps
    lr_scheduler.step()
    return avg_loss


@pytest.mark.dist
@pytest.mark.skip("This test should be invoked by test.sh in the same folder as it runs on multiple gpus")
def test_2d_parallel_vision_transformer():
    # init dist
    engine, train_dataloader, test_dataloader = colossalai.initialize(CONFIG_PATH)
    lr_scheduler = build_lr_scheduler(gpc.config.lr_scheduler, engine.optimizer)
    logger = get_global_dist_logger()

    logger.info('start training')
    for epoch in range(gpc.config.num_epochs):
        train_loss = train(engine, train_dataloader, lr_scheduler)

        logger.info(f'epoch {epoch} - train loss: {train_loss}')

        if epoch % 2 == 0:
            correct_sum, total_sum, eval_loss = eval(engine, test_dataloader)
            logger.info(
                f'epoch {epoch} - eval loss: {eval_loss}, total: {total_sum}, '
                f'correct: {correct_sum}, acc: {correct_sum / total_sum}')


if __name__ == '__main__':
    test_2d_parallel_vision_transformer()
