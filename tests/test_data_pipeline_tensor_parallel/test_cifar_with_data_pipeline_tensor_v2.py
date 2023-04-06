import os
from pathlib import Path

import pytest
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

import colossalai
from colossalai.amp import AMP_TYPE
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine.schedule._pipeline_schedule_v2 import PipelineScheduleV2
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import CrossEntropyLoss
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.pipeline.pipelinable import PipelinableContext
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.trainer import Trainer, hooks
from colossalai.utils import get_dataloader

disable_existing_loggers()
BATCH_SIZE = 4
NUM_EPOCHS = 10
WARMUP_EPOCHS = 5
CONFIG = dict(NUM_MICRO_BATCHES=2,
              parallel=dict(pipeline=2, tensor=dict(size=1, mode='1d')),
              fp16=dict(mode=AMP_TYPE.NAIVE),
              gradient_accumulation=2)


def run_trainer(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    disable_existing_loggers()
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

    engine._schedule = PipelineScheduleV2(num_microbatches=gpc.config.NUM_MICRO_BATCHES)

    logger = get_dist_logger()

    trainer = Trainer(engine=engine, logger=logger)

    hook_list = [
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
    ]

    trainer.fit(train_dataloader=train_dataloader,
                max_steps=2,
                epochs=NUM_EPOCHS,
                hooks=hook_list,
                display_progress=True)


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_hybrid_parallel():
    spawn(run_trainer, 2)
    disable_existing_loggers()


if __name__ == '__main__':
    test_hybrid_parallel()
