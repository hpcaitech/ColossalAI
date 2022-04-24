#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import transforms, datasets

import colossalai
from colossalai.context import ParallelMode, Config
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader, free_port
from colossalai.testing import rerun_if_address_is_in_use
from torchvision import transforms

CONFIG = Config(
    dict(
        train_data=dict(
            dataset=dict(
                type='CIFAR10',
                root=Path(os.environ['DATA']),
                train=True,
                download=True,
            ),
            dataloader=dict(num_workers=2, batch_size=2, shuffle=True),
        ),
        parallel=dict(
            pipeline=dict(size=1),
            tensor=dict(size=1, mode=None),
        ),
        seed=1024,
    ))


def run_data_sampler(rank, world_size, port):
    dist_args = dict(config=CONFIG, rank=rank, world_size=world_size, backend='gloo', port=port, host='localhost')
    colossalai.launch(**dist_args)

    # build dataset
    transform_pipeline = [transforms.ToTensor(), transforms.RandomCrop(size=32, padding=4)]
    transform_pipeline = transforms.Compose(transform_pipeline)
    dataset = datasets.CIFAR10(root=Path(os.environ['DATA']), train=True, download=True, transform=transform_pipeline)

    # build dataloader
    dataloader = get_dataloader(dataset, batch_size=8, add_sampler=False)

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
    torch.cuda.empty_cache()


@pytest.mark.cpu
@rerun_if_address_is_in_use()
def test_data_sampler():
    world_size = 4
    test_func = partial(run_data_sampler, world_size=world_size, port=free_port())
    mp.spawn(test_func, nprocs=world_size)


if __name__ == '__main__':
    test_data_sampler()
