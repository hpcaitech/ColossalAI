# referenced from Megatron and used to testify communication

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

import colossalai
from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.initialize import launch
from colossalai.legacy.utils import get_dataloader, print_rank_0
from colossalai.testing import rerun_if_address_is_in_use, spawn

BATCH_SIZE = 8

CONFIG = dict(NUM_MICRO_BATCHES=2, parallel=dict(pipeline=dict(size=2), tensor=dict(size=1, mode=None)))


def run_schedule(rank, world_size, port):
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    # build model
    model = resnet18(num_classes=10)

    if gpc.get_local_rank(ParallelMode.PIPELINE) == 0:
        model = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2)
    elif gpc.get_local_rank(ParallelMode.PIPELINE) == 1:

        class Flatten(nn.Module):
            def forward(self, x):
                return torch.flatten(x, 1)

        model = nn.Sequential(model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)

    print_rank_0("model is created")

    train_dataset = CIFAR10(
        root=Path(os.environ["DATA"]),
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]
        ),
    )

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        shuffle=True,
        add_sampler=True,
        batch_size=BATCH_SIZE,
        pin_memory=True,
    )

    # build criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    # initialize
    engine, train_dataloader, _, _ = colossalai.legacy.initialize(model, optimizer, criterion, train_dataloader)

    # build pipeline schedule
    schedule = engine.schedule

    # run schedule
    data_iter = iter(train_dataloader)
    schedule.forward_backward_step(engine, data_iter)

    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_pipeline_schedule():
    world_size = 2
    spawn(run_schedule, world_size)


if __name__ == "__main__":
    test_pipeline_schedule()
