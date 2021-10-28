#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# adpated from torch.utils.data.DistributedSampler

import math
from typing import TypeVar, Iterator

import torch
from torch.utils.data import Sampler, Dataset

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.registry import SAMPLERS

T_co = TypeVar('T_co', covariant=True)


@SAMPLERS.register_module
class DataParallelSampler(Sampler):
    """A data sampler for distributed data parallelism

    :param dataset: a Dataset instance
    :type dataset: torch.utils.data.Dataset
    :param shuffle: whether to shuffle data, defaults to False
    :type shuffle: bool, optional
    :param seed: the random seed, defaults to 0
    :type seed: int, optional
    :param drop_last: set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller, defaults to False
    :type drop_last: bool, optional
    """

    def __init__(self,
                 dataset: Dataset,
                 shuffle: bool = False,
                 seed: int = 0,
                 drop_last: bool = False) -> None:
        self.dataset = dataset
        self.num_replicas = gpc.get_world_size(ParallelMode.DATA)
        self.rank = gpc.get_local_rank(ParallelMode.DATA)
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        # type: ignore[arg-type]
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / \
                self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(
                len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size /
                            len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        :param epoch: Epoch number.
        :type epoch: int
        """
        self.epoch = epoch
