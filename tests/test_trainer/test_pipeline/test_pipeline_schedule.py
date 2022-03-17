# referenced from Megatron and used to testify communication

import os
import os.path as osp
from functools import partial
from pathlib import Path

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.builder import build_pipeline_model_from_cfg
from colossalai.core import global_context as gpc
from colossalai.engine.schedule import PipelineSchedule
from colossalai.initialize import launch
from colossalai.utils import free_port, get_dataloader, print_rank_0
from torchvision import transforms
from torchvision.datasets import CIFAR10

BATCH_SIZE = 4
NUM_MICRO = 2

DIR_PATH = osp.dirname(osp.realpath(__file__))
CONFIG_PATH = osp.join(DIR_PATH, './resnet_config.py')


def run_schedule(rank, world_size, port):
    launch(config=CONFIG_PATH, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    # build model
    model = build_pipeline_model_from_cfg(gpc.config.model, 1)
    print_rank_0('model is created')

    train_dataset = CIFAR10(root=Path(os.environ['DATA']),
                            download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                            ]))

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
    engine, train_dataloader, _, _ = colossalai.initialize(model, optimizer, criterion, train_dataloader)

    # build pipeline schedule
    schedule = PipelineSchedule(num_microbatches=NUM_MICRO)

    # run schedule
    data_iter = iter(train_dataloader)
    schedule.forward_backward_step(engine, data_iter)

    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
def test_pipeline_schedule():
    world_size = 4
    run_func = partial(run_schedule, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_pipeline_schedule()
