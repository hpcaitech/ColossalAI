import os

from functools import partial
from pathlib import Path

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.amp import AMP_TYPE
from colossalai.trainer import Trainer, hooks
from colossalai.context import ParallelMode
from colossalai.testing import rerun_if_address_is_in_use, skip_if_not_enough_gpus
from colossalai.utils import free_port
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import CrossEntropyLoss
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.utils import get_dataloader
from colossalai.pipeline.pipelinable import PipelinableContext
from torchvision.datasets import CIFAR10
from torchvision import transforms

BATCH_SIZE = 4
NUM_EPOCHS = 60
WARMUP_EPOCHS = 5
CONFIG = dict(NUM_MICRO_BATCHES=2,
              parallel=dict(pipeline=2, tensor=dict(size=2, mode='1d')),
              fp16=dict(mode=AMP_TYPE.NAIVE),
              gradient_accumulation=2)


def run_trainer(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    logger = get_dist_logger()

    # get logger
    logger = get_dist_logger()

    pipelinable = PipelinableContext()
    try:
        from titans.model.vit import vit_tiny_patch4_32
    except ImportError:
        logger.warning('skip the test_cifar_with_data_pipeline_tensor test because titan is not installed')
        logger.warning('please install titan from https://github.com/hpcaitech/Titans')
        return
    with pipelinable:
        model = vit_tiny_patch4_32()
    pipelinable.to_layer_list()
    pipelinable.policy = "uniform"
    model = pipelinable.partition(1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))

    # craete dataloaders
    root = Path(os.environ['DATA'])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, pad_if_needed=True),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform_train)
    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True)

    # create loss function
    criterion = CrossEntropyLoss(label_smoothing=0.1)

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)

    # create lr scheduler
    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer, total_steps=NUM_EPOCHS, warmup_steps=WARMUP_EPOCHS)

    # intiailize
    engine, train_dataloader, *_ = colossalai.initialize(model=model,
                                                         optimizer=optimizer,
                                                         criterion=criterion,
                                                         train_dataloader=train_dataloader)

    logger = get_dist_logger()

    trainer = Trainer(engine=engine, logger=logger)

    hook_list = [
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
    ]

    trainer.fit(train_dataloader=train_dataloader,
                epochs=NUM_EPOCHS,
                max_steps=2,
                hooks=hook_list,
                display_progress=True)


@pytest.mark.dist
@skip_if_not_enough_gpus(min_gpus=8)
@rerun_if_address_is_in_use()
def test_hybrid_parallel():
    world_size = 8
    run_func = partial(run_trainer, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_hybrid_parallel()
