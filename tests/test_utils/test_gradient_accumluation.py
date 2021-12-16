import colossalai
import os
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from functools import partial
from pathlib import Path
from torchvision import transforms
from torch.optim import Adam
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import report_memory_usage, get_dataloader
from colossalai.initialize import get_default_parser
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10


# Config
BATCH_SIZE = 16
IMG_SIZE = 224
NUM_CLASSES = 10

CONFIG = dict(
    parallel=dict(
        pipeline=dict(size=1),
        tensor=dict(size=1, mode=None)
    ),
    clip_grad_norm=1.0,
    gradient_accumulation=4
)


def run_no_pipeline(rank, world_size):

    # init dist env
    colossalai.launch(
        config=CONFIG,
        rank=rank,
        world_size=world_size,
        host='localhost',
        port=29500,
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
                                      pin_memory=True,
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
    param_track = []
    grad_track = []
    next(model.parameters()).retain_grad()

    engine.train()
    step = 0
    for img, label in train_dataloader:
        engine.zero_grad()
        img = img.cuda()
        label = label.cuda()
        output = engine(img)
        loss = engine.criterion(output, label)
        engine.backward(loss)
        engine.step()

        # check
        param_track.append(next(model.parameters())[0].clone())
        grad_track.append(next(model.parameters()).grad[0].clone())
        step += 1
        if step == CONFIG['gradient_accumulation']:
            break

    assert not torch.all(grad_track[0] == grad_track[-1]), 'grad should be different in different iterations'
    assert torch.all(param_track[0] == param_track[1]) and not torch.all(param_track[0] == param_track[-1]), \
        'param should be the same in the first few iterations and only changed in the last iteration'

    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_engine():
    world_size = 4
    func = partial(run_no_pipeline, world_size=world_size)
    mp.spawn(func, nprocs=world_size)


if __name__ == '__main__':
    test_engine()
