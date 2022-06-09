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
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import CrossEntropyLoss
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.utils import is_using_pp, get_dataloader
from colossalai.utils.model.pipelinable import PipelinableContext
from tqdm import tqdm

from titans.dataloader.cifar10 import build_cifar
from titans.model.vit import vit_tiny_patch4_32

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
    with pipelinable:
        model = vit_tiny_patch4_32()
    pipelinable.to_layer_list()
    pipelinable.load_policy("uniform")
    model = pipelinable.partition(1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))

    # craete dataloaders
    root = Path(os.environ['DATA'])
    train_dataloader, test_dataloader = build_cifar(BATCH_SIZE, root, pad_if_needed=True, crop=32, resize=32)

    # create loss function
    criterion = CrossEntropyLoss(label_smoothing=0.1)

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)

    # create lr scheduler
    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer, total_steps=NUM_EPOCHS, warmup_steps=WARMUP_EPOCHS)

    # intiailize
    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model=model,
                                                                         optimizer=optimizer,
                                                                         criterion=criterion,
                                                                         train_dataloader=train_dataloader,
                                                                         test_dataloader=test_dataloader)

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
@pytest.mark.skip("This test requires 8 GPUs to execute")
@rerun_if_address_is_in_use()
def test_hybrid_parallel():
    world_size = 8
    run_func = partial(run_trainer, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_hybrid_parallel()
