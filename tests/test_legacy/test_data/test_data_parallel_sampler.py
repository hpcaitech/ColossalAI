#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from pathlib import Path

import torch
import torch.distributed as dist
from torchvision import datasets, transforms

import colossalai
from colossalai.context import Config
from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.utils import get_dataloader
from colossalai.testing import rerun_if_address_is_in_use, spawn

CONFIG = Config(
    dict(
        parallel=dict(
            pipeline=dict(size=1),
            tensor=dict(size=1, mode=None),
        ),
        seed=1024,
    )
)


def run_data_sampler(rank, world_size, port):
    dist_args = dict(config=CONFIG, rank=rank, world_size=world_size, backend="gloo", port=port, host="localhost")
    colossalai.legacy.launch(**dist_args)
    print("finished initialization")

    # build dataset
    transform_pipeline = [transforms.ToTensor()]
    transform_pipeline = transforms.Compose(transform_pipeline)
    dataset = datasets.CIFAR10(root=Path(os.environ["DATA"]), train=True, download=True, transform=transform_pipeline)

    # build dataloader
    dataloader = get_dataloader(dataset, batch_size=8, add_sampler=True)

    data_iter = iter(dataloader)
    img, label = data_iter.next()
    img = img[0]

    if gpc.get_local_rank(ParallelMode.DATA) != 0:
        img_to_compare = img.clone()
    else:
        img_to_compare = img
    dist.broadcast(img_to_compare, src=0, group=gpc.get_group(ParallelMode.DATA))

    if gpc.get_local_rank(ParallelMode.DATA) != 0:
        assert not torch.equal(
            img, img_to_compare
        ), "Same image was distributed across ranks but expected it to be different"
    torch.cuda.empty_cache()


@rerun_if_address_is_in_use()
def test_data_sampler():
    spawn(run_data_sampler, 4)


if __name__ == "__main__":
    test_data_sampler()
