#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from pathlib import Path

import pytest
import torch.autograd

import colossalai
import torch
from colossalai.initialize import get_default_parser
from colossalai.builder import build_model
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import get_dataloader
from colossalai.nn.layer._parallel_utilities import _gather
from colossalai.nn import CrossEntropyLoss2D
from torchvision import transforms
from torchvision.datasets import CIFAR10
from components import *

level = os.environ['LEVEL']
CONFIG_PATH = Path(__file__).parent.parent.joinpath(f'configs/vit_2d_zero{level}.py')


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
    parser = get_default_parser()
    args = parser.parse_args()
    colossalai.launch(
        config=CONFIG_PATH,
        rank=args.rank,
        world_size=args.world_size,
        host=args.host,
        port=args.port,
        backend=args.backend
    )

    # build model
    model = build_model(model_cfg)
    model.build_from_cfg()

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss2D()

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


if __name__ == '__main__':
    test_2d_parallel_vision_transformer()
