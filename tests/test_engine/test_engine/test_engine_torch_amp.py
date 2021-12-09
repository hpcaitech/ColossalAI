import colossalai
import os
import pytest
import torch
import os.path as osp
from pathlib import Path
import torch.nn as nn

from torchvision import transforms
from torch.optim import Adam
from colossalai.core import global_context as gpc
from colossalai.amp import AMP_TYPE
from colossalai.logging import get_dist_logger
from colossalai.utils import report_memory_usage, get_dataloader
from colossalai.initialize import get_default_parser
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10


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
    fp16=dict(mode=AMP_TYPE.TORCH),
    clip_grad_norm=1.0
)


def run_no_pipeline():
    parser = get_default_parser()
    args = parser.parse_args()

    # init dist env
    colossalai.launch(
        config=CONFIG,
        rank=args.rank,
        world_size=args.world_size,
        host=args.host,
        port=args.port,
        backend=args.backend
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
                                      num_workers=1,
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


@pytest.mark.skip("This test should be invoked using the test.sh provided")
@pytest.mark.dist
def test_engine():
    run_no_pipeline()


if __name__ == '__main__':
    test_engine()
