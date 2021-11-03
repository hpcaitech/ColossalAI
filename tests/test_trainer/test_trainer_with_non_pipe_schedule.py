import os
from functools import partial
from pathlib import Path

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from colossalai.amp.amp_type import AMP_TYPE
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.trainer import Trainer
from colossalai.utils import MultiTimer, get_dataloader
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

BATCH_SIZE = 16
IMG_SIZE = 32
NUM_EPOCHS = 200

CONFIG = dict(
    # Config
    fp16=dict(mode=AMP_TYPE.TORCH))


def run_trainer_no_pipeline(rank, world_size):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=29930, backend='nccl')

    # build model
    model = resnet18(num_classes=10)

    # build dataloaders
    train_dataset = CIFAR10(root=Path(os.environ['DATA']),
                            download=True,
                            transform=transforms.Compose([
                                transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                            ]))

    test_dataset = CIFAR10(root=Path(os.environ['DATA']),
                           train=False,
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

    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=BATCH_SIZE, pin_memory=True, drop_last=True)

    # build optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    engine, train_dataloader, *args = colossalai.initialize(model=model,
                                                            optimizer=optimizer,
                                                            criterion=criterion,
                                                            train_dataloader=train_dataloader)

    logger = get_dist_logger()
    logger.info("engine is built", ranks=[0])

    timer = MultiTimer()
    trainer = Trainer(engine=engine, logger=logger, timer=timer)
    logger.info("trainer is built", ranks=[0])

    logger.info("start training", ranks=[0])
    trainer.fit(train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                epochs=NUM_EPOCHS,
                max_steps=100,
                display_progress=True,
                test_interval=5)
    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_trainer_no_pipeline():
    world_size = 4
    run_func = partial(run_trainer_no_pipeline, world_size=world_size)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_trainer_no_pipeline()
