#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def test_cifar10_dataset():
    # build transform
    transform_pipeline = [transforms.ToTensor()]
    transform_pipeline = transforms.Compose(transform_pipeline)

    # build dataset
    dataset = datasets.CIFAR10(root=Path(os.environ["DATA"]), train=True, download=True, transform=transform_pipeline)

    # build dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)
    data_iter = iter(dataloader)
    img, label = data_iter.next()


if __name__ == "__main__":
    test_cifar10_dataset()
