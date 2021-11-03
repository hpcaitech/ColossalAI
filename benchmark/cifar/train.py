#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import time

import colossalai
from colossalai.engine import schedule
import torch
import torchvision
from colossalai.builder import *
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import Accuracy, CrossEntropyLoss
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.trainer import Trainer
from colossalai.trainer.hooks import (AccuracyHook, LogMemoryByEpochHook, LogMetricByEpochHook, LogTimingByEpochHook,
                                      LossHook, LRSchedulerHook, ThroughputHook)
from colossalai.utils import MultiTimer, get_dataloader
from model_zoo.vit import vit_lite_7_patch4_32
from torchvision import transforms
from tqdm import tqdm

DATASET_PATH = str(os.environ['DATA'])


def build_cifar(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=DATASET_PATH,
                                                 train=True,
                                                 download=True,
                                                 transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, transform=transform_test)
    train_dataloader = get_dataloader(dataset=train_dataset,
                                      shuffle=True,
                                      batch_size=batch_size,
                                      num_workers=4,
                                      pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)
    return train_dataloader, test_dataloader


def train_epoch(engine, schedule, train_dataloader, epoch: int = None):
    # set training state
    engine.train()
    data_iter = iter(train_dataloader)
    progress = range(len(train_dataloader))
    if gpc.get_global_rank() == 0:
        progress = tqdm(progress, desc=f'[Epoch {epoch} train]')

    # metric measured by bian zhengda
    train_loss = 0
    batch_cnt = 0
    num_samples = 0
    ######
    for i in progress:
        # metric measured by bian zhengda
        cur_lr = engine.optimizer.param_groups[0]['lr']
        ######

        # run 1 training step
        batch_start = time.time()
        engine.zero_grad()
        _, label, loss = schedule.forward_backward_step(engine, data_iter, forward_only=False, return_loss=True)
        engine.step()
        batch_end = time.time()

        # metric measured by bian zhengda
        if gpc.get_global_rank() == 0:
            if isinstance(label, (tuple, list)):
                batch_size = label[0].size(0)
            else:
                batch_size = label.size(0)
            batch_size *= gpc.data_parallel_size
            train_loss += loss.item()
            num_samples += batch_size
            batch_cnt += 1
            batch_time = batch_end - batch_start
            print_features = dict(lr='%g' % cur_lr,
                                  loss='%.3f' % (train_loss / (i + 1)),
                                  throughput='%.3f (samples/sec)' % (batch_size / (batch_time + 1e-12)))
            progress.set_postfix(**print_features)
        ######


def train_cifar():
    args = colossalai.get_default_parser().parse_args()
    colossalai.launch_from_torch(config=args.config)
    # colossalai.launch(config=args.config,
    #                   rank=args.rank,
    #                   world_size=args.world_size,
    #                   local_rank=args.local_rank,
    #                   host=args.host,
    #                   port=args.port)
    logger = get_dist_logger()
    if hasattr(gpc.config, 'log_path'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.log_path
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    tp = gpc.config.parallel.tensor.mode

    model = vit_lite_7_patch4_32(tensor_parallel=tp)

    train_dataloader, test_dataloader = build_cifar(gpc.config.BATCH_SIZE // gpc.data_parallel_size)

    criterion = CrossEntropyLoss(label_smoothing=0.1, tensor_parallel=tp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)

    steps_per_epoch = len(train_dataloader) // gpc.config.gradient_accumulation

    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,
                                           total_steps=gpc.config.num_epochs * steps_per_epoch,
                                           warmup_steps=gpc.config.warmup_epochs * steps_per_epoch)

    engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(model=model,
                                                                                    optimizer=optimizer,
                                                                                    criterion=criterion,
                                                                                    train_dataloader=train_dataloader,
                                                                                    test_dataloader=test_dataloader,
                                                                                    lr_scheduler=lr_scheduler)

    logger.info("Engine is built", ranks=[0])

    # sched = schedule.NonPipelineSchedule()
    # for epoch in range(gpc.config.num_epochs):
    #     train_epoch(engine, sched, train_dataloader, epoch)

    timer = MultiTimer()

    trainer = Trainer(engine=engine, logger=logger, timer=timer)
    logger.info("Trainer is built", ranks=[0])

    hooks = [
        LogMetricByEpochHook(logger=logger),
        # LogTimingByEpochHook(timer=timer, logger=logger),
        # LogMemoryByEpochHook(logger=logger),
        AccuracyHook(accuracy_func=Accuracy(tensor_parallel=tp)),
        LossHook(),
        ThroughputHook(),
        LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False)
    ]

    logger.info("Train start", ranks=[0])
    trainer.fit(train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                epochs=gpc.config.num_epochs,
                hooks=hooks,
                display_progress=True,
                test_interval=1)


if __name__ == '__main__':
    train_cifar()
