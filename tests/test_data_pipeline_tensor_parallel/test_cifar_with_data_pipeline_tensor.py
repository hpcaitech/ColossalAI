import os
from functools import partial
from pathlib import Path

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.amp.amp_type import AMP_TYPE
from colossalai.builder import build_pipeline_model
from colossalai.engine.schedule import PipelineSchedule
from colossalai.logging import get_dist_logger
from colossalai.nn import LinearWarmupLR
from colossalai.nn.loss import CrossEntropyLoss
from colossalai.trainer import Trainer, hooks
from colossalai.utils import free_port, get_dataloader
from colossalai.utils.gradient_accumulation import GradAccumLrSchedulerByStep
from colossalai.testing import rerun_on_exception
from model_zoo.vit import vit_tiny_patch4_32
from torchvision import transforms
from torchvision.datasets import CIFAR10

BATCH_SIZE = 4
NUM_EPOCHS = 60
WARMUP_EPOCHS = 5
CONFIG = dict(parallel=dict(pipeline=2, tensor=dict(size=2, mode='1d')),
              fp16=dict(mode=AMP_TYPE.NAIVE),
              gradient_accumulation=2)


def run_trainer(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    logger = get_dist_logger()

    model = vit_tiny_patch4_32()
    pipe_model = build_pipeline_model(model.layers, num_chunks=1)

    # build dataloaders
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(root=Path(os.environ['DATA']), train=True, download=True, transform=transform_train)
    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True)

    # build criterion
    criterion = CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(pipe_model.parameters(), lr=0.001, weight_decay=0)

    # lr_scheduler
    steps_per_epoch = GradAccumLrSchedulerByStep.compute_effective_steps_per_epoch(train_dataloader, accumulate_size=2)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS
    lr_scheduler = LinearWarmupLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(pipe_model,
                                                                      optimizer,
                                                                      criterion,
                                                                      train_dataloader,
                                                                      lr_scheduler=lr_scheduler)

    schedule = PipelineSchedule(num_microbatches=2)
    logger = get_dist_logger()

    trainer = Trainer(engine=engine, logger=logger, schedule=schedule)

    hook_list = [
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
    ]

    trainer.fit(train_dataloader=train_dataloader,
                epochs=NUM_EPOCHS,
                max_steps=2,
                hooks=hook_list,
                display_progress=True)


@pytest.mark.dist
@rerun_on_exception(exception_type=mp.ProcessRaisedException, pattern=".*Address already in use.*")
def test_hybrid_parallel():
    world_size = 8
    run_func = partial(run_trainer, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_hybrid_parallel()
