import random
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from colossalai.booster.interface import OptimizerWrapper

from .plugin_base import Plugin

__all__ = ['TorchDDPPlugin']


class TorchDDPPlugin(Plugin):
    """
    Plugin for PyTorch DDP.

    Example:
        >>> from colossalai.booster import Booster
        >>> from colossalai.booster.plugin import TorchDDPPlugin
        >>>
        >>> model, train_dataset, optimizer, criterion = ...
        >>> plugin = TorchDDPPlugin()

        >>> train_dataloader = plugin.prepare_train_dataloader(train_dataset, batch_size=8)
        >>> booster = Booster(plugin=plugin)
        >>> model, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)

    Args:
        broadcast_buffers (bool, optional): Whether to broadcast buffers in the beginning of training. Defaults to True.
        bucket_cap_mb (int, optional): The bucket size in MB. Defaults to 25.
        find_unused_parameters (bool, optional): Whether to find unused parameters. Defaults to False.
        check_reduction (bool, optional): Whether to check reduction. Defaults to False.
        gradient_as_bucket_view (bool, optional): Whether to use gradient as bucket view. Defaults to False.
        static_graph (bool, optional): Whether to use static graph. Defaults to False.
    """

    def __init__(self,
                 broadcast_buffers: bool = True,
                 bucket_cap_mb: int = 25,
                 find_unused_parameters: bool = False,
                 check_reduction: bool = False,
                 gradient_as_bucket_view: bool = False,
                 static_graph: bool = False) -> None:

        assert dist.is_initialized(
        ), 'torch.distributed is not initialized, please use colossalai.launch to create the distributed environment'
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.ddp_kwargs = dict(broadcast_buffers=broadcast_buffers,
                               bucket_cap_mb=bucket_cap_mb,
                               find_unused_parameters=find_unused_parameters,
                               check_reduction=check_reduction,
                               gradient_as_bucket_view=gradient_as_bucket_view,
                               static_graph=static_graph)

    def support_no_sync(self) -> bool:
        return True

    def control_precision(self) -> bool:
        return False

    def supported_precisions(self) -> List[str]:
        return ['fp16', 'fp16_apex', 'bf16', 'fp8']

    def control_device(self) -> bool:
        return True

    def supported_devices(self) -> List[str]:
        return ['cuda']

    def prepare_train_dataloader(self,
                                 dataset,
                                 batch_size,
                                 shuffle=False,
                                 seed=1024,
                                 drop_last=False,
                                 pin_memory=False,
                                 num_workers=0,
                                 **kwargs):
        r"""
        Prepare a dataloader for distributed training. The dataloader will be wrapped by
        `torch.utils.data.DataLoader` and `torch.utils.data.DistributedSampler`.

        Note:
            1. Evaluation datasets should not be passed to this function.

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
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle)

        # Deterministic dataloader
        def seed_worker(worker_id):
            worker_seed = seed
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(dataset,
                          batch_size=batch_size,
                          sampler=sampler,
                          worker_init_fn=seed_worker,
                          drop_last=drop_last,
                          pin_memory=pin_memory,
                          num_workers=num_workers,
                          **_kwargs)

    def configure(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Callable = None,
        dataloader: DataLoader = None,
        lr_scheduler: LRScheduler = None,
    ) -> Tuple[Union[nn.Module, OptimizerWrapper, LRScheduler, DataLoader]]:
        # cast model to cuda
        model = model.cuda()

        # wrap the model with PyTorch DDP
        model = DDP(model, **self.ddp_kwargs)

        if not isinstance(optimizer, OptimizerWrapper):
            optimizer = OptimizerWrapper(optimizer)

        return model, optimizer, criterion, dataloader, lr_scheduler
