#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from functools import partial
from pathlib import Path

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.amp.naive_amp import NaiveAMPModel
from colossalai.builder.builder import build_gradient_handler
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine import Engine
from colossalai.utils import free_port, get_dataloader
from colossalai.utils.cuda import get_current_device
from colossalai.zero.zero_redundancy_optimizer_level_1 import \
    ZeroRedundancyOptimizer_Level_1
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

BATCH_SIZE = 16
IMG_SIZE = 224

CONFIG = dict(
    fp16=dict(
        mode=None,
    ),
    zero=dict(
        level=1,
        cpu_offload=True,
        verbose=False,
        partition_grads=False,
    ),
    parallel=dict(
        pipeline=dict(size=1),
        tensor=dict(size=1, mode=None)
    )
)


def run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG,
                      rank=rank,
                      world_size=world_size,
                      host='localhost',
                      port=port,
                      backend='nccl')

    # build model
    model = resnet18(num_classes=10)

    # build dataloader# build dataloaders
    train_dataset = CIFAR10(
        root=Path(os.environ['DATA']),
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )
    )
    train_dataloader = get_dataloader(dataset=train_dataset,
                                      shuffle=True,
                                      batch_size=BATCH_SIZE,
                                      pin_memory=True,
                                      drop_last=True)

    # build optimizer and loss
    # optimizer = build_optimizer(global_context.config.optimizer, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # model.to(get_current_device())
    # model = NaiveAMPModel(model, output_to_fp32=False)
    # optimizer = ZeroRedundancyOptimizer_Level_1(
    #     optimizer, partition_grads=False, overlap_comm=False, cpu_offload=True, dp_process_group=gpc.get_group(ParallelMode.DATA))

    # engine = Engine(
    #     model=model,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     gradient_handlers=[build_gradient_handler(dict(type='ZeROGradientHandler'), model, optimizer)],
    #     clip_grad_norm=0.0
    # )
    engine, train_dataloader, *args = colossalai.initialize(model=model,
                                                            optimizer=optimizer,
                                                            criterion=criterion,
                                                            train_dataloader=train_dataloader)

    # train
    model.train()
    for idx, (data, label) in enumerate(train_dataloader):
        engine.zero_grad()
        data = data.cuda()
        label = label.cuda()

        output = engine(data)
        loss = engine.criterion(output, label)

        engine.backward(loss)
        engine.step()
        break

    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_zero_level_1():
    world_size = 4
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_zero_level_1()
