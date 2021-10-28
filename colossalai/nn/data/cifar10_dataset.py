#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.distributed as dist
from torchvision.datasets import CIFAR10

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module
class CIFAR10Dataset(BaseDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    :param transform_pipeline: A list of functions' config, which takes in an PIL image
            and returns a transformed version
    :type transform_pipeline: list
    """

    def __init__(self, transform_pipeline: list, *args, **kwargs):
        super().__init__(transform_pipeline)
        if gpc.is_initialized(ParallelMode.GLOBAL) and gpc.get_global_rank() != 0:
            dist.barrier()
        self._dataset = CIFAR10(transform=self._transform_pipeline,
                                *args,
                                **kwargs)
        if gpc.is_initialized(ParallelMode.GLOBAL) and gpc.get_global_rank() == 0:
            dist.barrier()

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        """

        :param item: Index
        :type item: int
        :return: ((image,), (target,)) where the type of target specified by target_type.
        :rtype: tuple
        """
        img, label = self._dataset.__getitem__(item)
        return (img,), (label,)
