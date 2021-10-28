#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from pathlib import Path

import pytest
import torch.autograd

import colossalai
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine import Engine
from colossalai.logging import get_global_dist_logger
from colossalai.nn.layer._parallel_utilities import _gather

CONFIG_PATH = Path(__file__).parent.parent.joinpath('configs/vit_2d.py')


def eval(engine):
    engine.eval()
    accumulated_loss = 0
    correct_sum = 0
    total_sum = 0

    for i in range(engine.schedule.num_steps):
        output, label, loss = engine.step()

        if gpc.is_last_rank(ParallelMode.PIPELINE):
            # loss = sum(loss)
            accumulated_loss += loss.detach().cpu().numpy()

            output = _gather(
                output,
                ParallelMode.PARALLEL_2D_ROW,
                1
            )
            output = _gather(
                output,
                ParallelMode.PARALLEL_2D_COL,
                0,
            )
            output = torch.argmax(output, dim=-1)
            correct = torch.sum(label == output)
            correct_sum += correct
            total_sum += label.size(0)
    avg_loss = accumulated_loss / engine.schedule.num_steps
    return correct_sum, total_sum, avg_loss


def train(engine):
    engine.train()
    accumulated_loss = 0

    for i in range(engine.schedule.num_steps):
        output, label, loss = engine.step()

        if gpc.is_last_rank(ParallelMode.PIPELINE):
            accumulated_loss += loss.detach().cpu().numpy()
    avg_loss = accumulated_loss / engine.schedule.num_steps
    return avg_loss


@pytest.mark.dist
@pytest.mark.skip("This test should be invoked by test.sh in the same folder as it runs on multiple gpus")
def test_2d_parallel_vision_transformer():
    # init dist
    model, train_dataloader, test_dataloader, criterion, optimizer, schedule, lr_scheduler = colossalai.initialize(
        CONFIG_PATH)
    logger = get_global_dist_logger()

    engine = Engine(model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    criterion=criterion,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    schedule=schedule)

    for epoch in range(gpc.config.num_epochs):
        train_loss = train(engine)
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            logger.info(f'epoch {epoch} - train loss: {train_loss}')

        if epoch % 2 == 0:
            correct_sum, total_sum, eval_loss = eval(engine)
            if gpc.is_last_rank(ParallelMode.PIPELINE):
                logger.info(
                    f'epoch {epoch} - eval loss: {eval_loss}, total: {total_sum}, '
                    f'correct: {correct_sum}, acc: {correct_sum / total_sum}')


if __name__ == '__main__':
    test_2d_parallel_vision_transformer()
