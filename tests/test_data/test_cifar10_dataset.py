#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from colossalai.builder import build_dataset
from colossalai.context import Config

train_data = dict(
    dataset=dict(
        type='CIFAR10Dataset',
        root=Path(os.environ['DATA']),
        train=True,
        download=True,
        transform_pipeline=[
            dict(type='ToTensor'),
            dict(type='Normalize',
                 mean=(0.5, 0.5, 0.5),
                 std=(0.5, 0.5, 0.5))
        ]),
    dataloader=dict(batch_size=4, shuffle=True, num_workers=2)
)


@pytest.mark.cpu
def test_cifar10_dataset():
    global train_data
    config = Config(train_data)
    dataset = build_dataset(config.dataset)
    dataloader = DataLoader(dataset=dataset, **config.dataloader)
    data_iter = iter(dataloader)
    img, label = data_iter.next()

    assert isinstance(img, list) and isinstance(label, list), \
        f'expected the img and label to be list but got {type(img)} and {type(label)}'


if __name__ == '__main__':
    test_cifar10_dataset()
