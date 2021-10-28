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
from colossalai.builder import build_dataset, build_data_sampler
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
                dict(type='Normalize', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        ),
        dataloader=dict(
            num_workers=2,
            batch_size=8,
            sampler=dict(
                type='DataParallelSampler',
            )
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
        port='29503',
        host='localhost'
    )
    colossalai.init_dist(**dist_args)
    print('finished initialization')

    dataset = build_dataset(gpc.config.train_data.dataset)
    sampler_cfg = gpc.config.train_data.dataloader.pop('sampler')
    sampler = build_data_sampler(sampler_cfg, dataset)
    dataloader = DataLoader(dataset=dataset, sampler=sampler, **gpc.config.train_data.dataloader)
    data_iter = iter(dataloader)
    img, label = data_iter.next()
    img = img[0]

    if gpc.get_local_rank(ParallelMode.DATA) != 0:
        img_to_compare = img.clone()
    else:
        img_to_compare = img
    dist.broadcast(img_to_compare, src=0, group=gpc.get_group(ParallelMode.DATA))

    if gpc.get_local_rank(ParallelMode.DATA) != 0:
        assert not torch.equal(img,
                               img_to_compare), 'Same image was distributed across ranks but expected it to be different'


@pytest.mark.cpu
def test_data_sampler():
    world_size = 4
    test_func = partial(run_data_sampler, world_size=world_size)
    mp.spawn(test_func, nprocs=world_size)


if __name__ == '__main__':
    test_data_sampler()
