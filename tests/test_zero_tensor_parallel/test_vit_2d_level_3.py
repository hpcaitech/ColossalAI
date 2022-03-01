#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from functools import partial
from pathlib import Path

import colossalai
import pytest
import torch
import torch.autograd
import torch.multiprocessing as mp
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import CrossEntropyLoss
from colossalai.utils import free_port, get_dataloader
from model_zoo.vit import vit_lite_depth7_patch4_32
from torchvision import transforms
from torchvision.datasets import CIFAR10

from components import *

CONFIG = dict(parallel=dict(
    pipeline=dict(size=1),
    tensor=dict(size=4, mode='2d'),
),
              fp16=dict(mode=None, ),
              zero=dict(level=3))


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


def run_2d_parallel_vision_transformer_level_3(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    # build model
    model = vit_lite_depth7_patch4_32()

    # build dataloader# build dataloaders
    train_dataset = CIFAR10(root=Path(os.environ['DATA']),
                            download=True,
                            transform=transforms.Compose([
                                transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                            ]))
    train_dataloader = get_dataloader(dataset=train_dataset,
                                      shuffle=True,
                                      batch_size=BATCH_SIZE,
                                      pin_memory=True,
                                      drop_last=True)

    # build optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    engine, train_dataloader, *args = colossalai.initialize(model=model,
                                                            optimizer=optimizer,
                                                            criterion=criterion,
                                                            train_dataloader=train_dataloader)
    logger = get_dist_logger()

    logger.info('start training')
    engine.train()

    for img, label in train_dataloader:
        engine.zero_grad()
        img = img.cuda()
        label = label.cuda()
        out = engine(img)
        loss = engine.criterion(out, label)
        engine.backward(loss)
        engine.step()
        break

    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
@pytest.mark.skip(reason="This test should be refactored for the reconstructed zero")
def test_3d_vit_zero_level_3():
    world_size = 8
    run_func = partial(run_2d_parallel_vision_transformer_level_3, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_3d_vit_zero_level_3()
