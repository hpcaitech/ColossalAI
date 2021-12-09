#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import pytest
import torch

from pathlib import Path

import colossalai
from colossalai.initialize import get_default_parser
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10

BATCH_SIZE = 128
IMG_SIZE = 224
NUM_CLS = 1000

CONFIG = dict(
    fp16=dict(
        mode=None,
    ),
    zero=dict(
        # ==============
        # level 2 config
        # ==============
        # level=2,
        # cpu_offload=True,
        # verbose=False,

        # ==============
        # level 3 config
        # ==============
        level=3,
        verbose=False,
        offload_optimizer_config=dict(
            device='cpu',
            pin_memory=True,
            buffer_count=5,
            fast_init=False
        ),
        offload_param_config=dict(
            device='cpu',
            pin_memory=True,
            buffer_count=5,
            buffer_size=1e8,
            max_in_cpu=1e9
        )
    ),
    parallel=dict(
        pipeline=dict(size=1),
        tensor=dict(size=1, mode=None)
    )
)


def run_dist():
    parser = get_default_parser()
    args = parser.parse_args()

    colossalai.launch(config=CONFIG,
                         rank=args.rank,
                         world_size=args.world_size,
                         host=args.host,
                         port=args.port,
                         backend=args.backend)

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
                                      num_workers=1,
                                      pin_memory=True,
                                      drop_last=True)

    # build optimizer and loss
    # optimizer = build_optimizer(global_context.config.optimizer, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

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


@pytest.mark.skip("This test should be invoked manually using the script provided")
@pytest.mark.dist
def test_zero():
    run_dist()


if __name__ == '__main__':
    test_zero()
