#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
from pathlib import Path

import torch
from tqdm import tqdm

from colossalai import initialize
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine import Engine
from colossalai.logging import get_global_dist_logger
from colossalai.trainer import Trainer
from colossalai.trainer.metric import Accuracy3D
from colossalai.utils import print_rank_0

CONFIG_PATH = Path(__file__).parent.parent.joinpath('configs/vit_3d.py')


def _train_epoch(epoch, engine):
    logger = get_global_dist_logger()
    print_rank_0('[Epoch %d] training start' % (epoch), logger)
    engine.train()

    train_loss = 0
    batch_cnt = 0
    num_samples = 0
    now = time.time()
    epoch_start = now
    progress = range(engine.schedule.num_steps)
    if gpc.get_global_rank() == 0:
        progress = tqdm(progress, desc='[Epoch %d]' % epoch, miniters=1)
    for step in progress:
        cur_lr = engine.get_lr()

        _, targets, loss = engine.step()

        batch_size = targets[0].size(0)
        train_loss += loss.item()
        num_samples += batch_size
        batch_cnt += 1

        batch_time = time.time() - now
        now = time.time()
        if gpc.get_global_rank() == 0:
            print_features = dict(lr='%g' % cur_lr,
                                  loss='%.3f' % (train_loss / (step + 1)),
                                  throughput='%.3f (images/sec)' %
                                             (batch_size / (batch_time + 1e-12)))
            progress.set_postfix(**print_features)

    epoch_end = time.time()
    epoch_loss = train_loss / batch_cnt
    epoch_throughput = num_samples / (epoch_end - epoch_start + 1e-12)
    print_rank_0(
        '[Epoch %d] Loss: %.3f | Throughput: %.3f (samples/sec)' %
        (epoch, epoch_loss, epoch_throughput), logger)


def _eval(epoch, engine):
    logger = get_global_dist_logger()
    engine.eval()

    eval_loss = 0
    acc = Accuracy3D(True, ParallelMode.PARALLEL_3D_OUTPUT,
                     ParallelMode.PARALLEL_3D_WEIGHT)
    total = 0
    with torch.no_grad():
        for _ in range(engine.schedule.num_steps):
            outputs, targets, loss = engine.step()
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            if isinstance(targets, (list, tuple)):
                targets = targets[0]
            eval_loss += loss.item()
            acc.update(outputs, targets)
            total += targets.size(0)

        print_rank_0(
            '[Epoch %d] Evaluation loss: %.3f | Acc: %.3f%%' %
            (epoch, eval_loss / engine.schedule.num_steps,
             acc.get_accumulated_value() * 100), logger)


def train():
    model, train_dataloader, test_dataloader, criterion, \
    optimizer, schedule, lr_scheduler = initialize(CONFIG_PATH)

    logger = get_global_dist_logger()

    engine = Engine(model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    criterion=criterion,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    schedule=schedule)
    logger.info("Engine is built", ranks=[0])

    trainer = Trainer(engine=engine, hooks_cfg=gpc.config.hooks, verbose=True)
    logger.info("Trainer is built", ranks=[0])

    logger.info("Train start", ranks=[0])
    trainer.fit(train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                max_epochs=gpc.config.num_epochs,
                display_progress=True,
                test_interval=1)


if __name__ == '__main__':
    train()
