#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import transforms
from torch.utils.data import DataLoader

import colossalai
from colossalai.builder import build_dataset, build_transform
from colossalai.context import ParallelMode, Config
from colossalai.core import global_context as gpc
from colossalai.utils import free_port
from colossalai.testing import rerun_on_exception

CONFIG = Config(
    dict(
        train_data=dict(dataset=dict(
            type='CIFAR10',
            root=Path(os.environ['DATA']),
            train=True,
            download=True,
        ),
                        dataloader=dict(num_workers=2, batch_size=2, shuffle=True),
                        transform_pipeline=[
                            dict(type='ToTensor'),
                            dict(type='RandomCrop', size=32),
                            dict(type='Normalize', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                        ]),
        parallel=dict(
            pipeline=dict(size=1),
            tensor=dict(size=1, mode=None),
        ),
        seed=1024,
    ))


def run_data_sampler(rank, world_size, port):
    dist_args = dict(config=CONFIG, rank=rank, world_size=world_size, backend='gloo', port=port, host='localhost')
    colossalai.launch(**dist_args)

    dataset_cfg = gpc.config.train_data.dataset
    dataloader_cfg = gpc.config.train_data.dataloader
    transform_cfg = gpc.config.train_data.transform_pipeline

    # build transform
    transform_pipeline = [build_transform(cfg) for cfg in transform_cfg]
    transform_pipeline = transforms.Compose(transform_pipeline)
    dataset_cfg['transform'] = transform_pipeline

    # build dataset
    dataset = build_dataset(dataset_cfg)

    # build dataloader
    dataloader = DataLoader(dataset=dataset, **dataloader_cfg)

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
@rerun_on_exception(exception_type=mp.ProcessRaisedException, pattern=".*Address already in use.*")
def test_data_sampler():
    world_size = 4
    test_func = partial(run_data_sampler, world_size=world_size, port=free_port())
    mp.spawn(test_func, nprocs=world_size)


if __name__ == '__main__':
    test_data_sampler()
