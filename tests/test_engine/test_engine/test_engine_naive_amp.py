import colossalai
import os
import pytest
import torch
import os.path as osp
from pathlib import Path
import torch.nn as nn
import torch.multiprocessing as mp

from torchvision import transforms
from torch.optim import Adam
from colossalai.core import global_context as gpc
from colossalai.amp import AMP_TYPE
from colossalai.logging import get_dist_logger
from colossalai.utils import report_memory_usage, get_dataloader
from colossalai.initialize import get_default_parser
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from functools import partial


# Config
BATCH_SIZE = 128
IMG_SIZE = 224
DIM = 768
NUM_CLASSES = 10
NUM_ATTN_HEADS = 12

CONFIG = dict(
    parallel=dict(
        pipeline=dict(size=1),
        tensor=dict(size=1, mode=None)
    ),
    fp16=dict(
        mode=AMP_TYPE.NAIVE,
        clip_grad=1.0
    )
)


def run_engine(rank, world_size):
    # init dist env
    colossalai.launch(
        config=CONFIG,
        rank=rank,
        world_size=world_size,
        host='localhost',
        port=29911,
        backend='nccl'
    )

    # build model
    model = resnet18(num_classes=10)

    # build dataloaders
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
                                      drop_last=True)

    # build optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    engine, train_dataloader, *args = colossalai.initialize(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader
    )
    logger = get_dist_logger()
    rank = torch.distributed.get_rank()

    engine.train()
    for img, label in train_dataloader:
        engine.zero_grad()
        img = img.cuda()
        label = label.cuda()
        output = engine(img)
        loss = engine.criterion(output, label)
        engine.backward(loss)
        engine.step()
        break

    logger.info('Rank {} returns: {}'.format(rank, loss.item()))

    gpc.destroy()
    logger.info('Test engine finished')
    report_memory_usage("After testing")
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_engine():
    world_size = 4
    run_func = partial(run_engine, world_size=world_size)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_engine()
