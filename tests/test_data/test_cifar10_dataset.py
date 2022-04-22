#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from pathlib import Path

import pytest
from torchvision import transforms
from torch.utils.data import DataLoader

from colossalai.builder import build_dataset, build_transform
from colossalai.context import Config
from torchvision.transforms import ToTensor

TRAIN_DATA = dict(dataset=dict(type='CIFAR10', root=Path(os.environ['DATA']), train=True, download=True),
                  dataloader=dict(batch_size=4, shuffle=True, num_workers=2))


@pytest.mark.cpu
def test_cifar10_dataset():
    config = Config(TRAIN_DATA)
    dataset_cfg = config.dataset
    dataloader_cfg = config.dataloader
    transform_cfg = config.transform_pipeline

    # build transform
    transform_pipeline = [ToTensor()]
    transform_pipeline = transforms.Compose(transform_pipeline)
    dataset_cfg['transform'] = transform_pipeline

    # build dataset
    dataset = build_dataset(dataset_cfg)

    # build dataloader
    dataloader = DataLoader(dataset=dataset, **dataloader_cfg)
    data_iter = iter(dataloader)
    img, label = data_iter.next()


if __name__ == '__main__':
    test_cifar10_dataset()
