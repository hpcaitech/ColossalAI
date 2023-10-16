import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

import colossalai
from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.trainer import Trainer
from colossalai.legacy.utils import get_dataloader
from colossalai.logging import get_dist_logger
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils import MultiTimer

BATCH_SIZE = 4
IMG_SIZE = 32
NUM_EPOCHS = 200

CONFIG = dict(
    NUM_MICRO_BATCHES=2,
    parallel=dict(pipeline=2),
)


def run_trainer_with_pipeline(rank, world_size, port):
    colossalai.legacy.launch(
        config=CONFIG, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl"
    )

    # build model
    model = resnet18(num_classes=10)

    if gpc.get_local_rank(ParallelMode.PIPELINE) == 0:
        model = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2)
    elif gpc.get_local_rank(ParallelMode.PIPELINE) == 1:

        class Flatten(nn.Module):
            def forward(self, x):
                return torch.flatten(x, 1)

        model = nn.Sequential(model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)

    # build dataloaders
    train_dataset = CIFAR10(
        root=Path(os.environ["DATA"]),
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        ),
    )

    train_dataloader = get_dataloader(
        dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True, drop_last=True
    )

    # build optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    engine, train_dataloader, *args = colossalai.legacy.initialize(
        model=model, optimizer=optimizer, criterion=criterion, train_dataloader=train_dataloader
    )

    logger = get_dist_logger()
    logger.info("engine is built", ranks=[0])
    timer = MultiTimer()
    trainer = Trainer(engine=engine, logger=logger, timer=timer)
    logger.info("trainer is built", ranks=[0])

    logger.info("start training", ranks=[0])

    trainer.fit(
        train_dataloader=train_dataloader, epochs=NUM_EPOCHS, max_steps=3, display_progress=True, test_interval=5
    )
    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_trainer_with_pipeline():
    world_size = 4
    spawn(run_trainer_with_pipeline, world_size)


if __name__ == "__main__":
    test_trainer_with_pipeline()
