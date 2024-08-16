import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .plugin_base import Plugin


class DPPluginBase(Plugin):
    """This is a base class for all DP plugins. It sets up world size and rank, and provides data loader creation."""

    def __init__(self) -> None:
        super().__init__()
        assert (
            dist.is_initialized()
        ), "torch.distributed is not initialized, please use colossalai.launch to create the distributed environment"
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def prepare_dataloader(
        self,
        dataset,
        batch_size,
        shuffle=False,
        seed=1024,
        drop_last=False,
        pin_memory=False,
        num_workers=0,
        distributed_sampler_cls=None,
        **kwargs,
    ):
        r"""
        Prepare a dataloader for distributed training. The dataloader will be wrapped by
        `torch.utils.data.DataLoader` and `torch.utils.data.DistributedSampler`.


        Args:
            dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
            seed (int, optional): Random worker seed for sampling, defaults to 1024.
            add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
            drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
                is not divisible by the batch size. If False and the size of dataset is not divisible by
                the batch size, then the last batch will be smaller, defaults to False.
            pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
            num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
            kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                    `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.

        Returns:
            :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
        """
        _kwargs = kwargs.copy()
        distributed_sampler_cls = distributed_sampler_cls or DistributedSampler
        sampler = distributed_sampler_cls(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle)

        # Deterministic dataloader
        def seed_worker(worker_id):
            worker_seed = seed
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            worker_init_fn=seed_worker,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )
