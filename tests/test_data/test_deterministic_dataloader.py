#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from functools import partial
from pathlib import Path

import pytest
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import colossalai
from colossalai.builder import build_dataset
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc

CONFIG = dict(
    train_data=dict(
        dataset=dict(
            type='CIFAR10Dataset',
            root=Path(os.environ['DATA']),
            train=True,
            download=True,
            transform_pipeline=[
                dict(type='ToTensor'),
                dict(type='RandomCrop', size=32),
                dict(type='Normalize', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        ),
        dataloader=dict(
            num_workers=2,
            batch_size=2,
            shuffle=True
        )
    ),
    parallel=dict(
        pipeline=dict(size=1),
        tensor=dict(size=1, mode=None),
    ),
    seed=1024,
)


def run_data_sampler(local_rank, world_size):
    dist_args = dict(
        config=CONFIG,
        local_rank=local_rank,
        world_size=world_size,
        backend='gloo',
        port='29499',
        host='localhost'
    )
    colossalai.init_dist(**dist_args)
    gpc.set_seed()

    print('finished initialization')

    dataset = build_dataset(gpc.config.train_data.dataset)
    dataloader = DataLoader(dataset=dataset, **gpc.config.train_data.dataloader)
    data_iter = iter(dataloader)
    img, label = data_iter.next()
    img = img[0]

    if gpc.get_local_rank(ParallelMode.DATA) != 0:
        img_to_compare = img.clone()
    else:
        img_to_compare = img
    dist.broadcast(img_to_compare, src=0, group=gpc.get_group(ParallelMode.DATA))

    if gpc.get_local_rank(ParallelMode.DATA) != 0:
        # this is without sampler
        # this should be false if data parallel sampler to given to the dataloader
        assert torch.equal(img,
                           img_to_compare), 'Same image was distributed across ranks and expected it to be the same'


@pytest.mark.cpu
def test_data_sampler():
    world_size = 4
    test_func = partial(run_data_sampler, world_size=world_size)
    mp.spawn(test_func, nprocs=world_size)


if __name__ == '__main__':
    test_data_sampler()
