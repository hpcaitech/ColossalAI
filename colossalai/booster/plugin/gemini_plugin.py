import random
import warnings
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from colossalai.checkpoint_io import CheckpointIO, GeneralCheckpointIO
from colossalai.cluster import DistCoordinator
from colossalai.gemini.memory_tracer import MemStats
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.nn.parallel import GeminiDDP, zero_model_wrapper, zero_optim_wrapper
from colossalai.utils import get_current_device

from .plugin_base import Plugin

__all__ = ['TorchDDPPlugin']


class GeminiCheckpointIO(GeneralCheckpointIO):

    def __init__(self) -> None:
        super().__init__()
        self.coordinator = DistCoordinator()

    def load_unsharded_model(self, model: GeminiDDP, checkpoint: str, strict: bool = True):
        """
        Load model from checkpoint with automatic unwrapping.
        """
        # the model should be unwrapped in self.load_model via ModelWrapper.unwrap
        return super().load_unsharded_model(model, checkpoint, strict=strict)

    def save_unsharded_model(self, model: GeminiDDP, checkpoint: str):
        """
        Save model to checkpoint but only on master process.
        """
        # the model should be unwrapped in self.load_model via ModelWrapper.unwrap
        # as there is communication when get state dict, this must be called on all processes
        state_dict = model.state_dict(only_rank_0=True)
        if self.coordinator.is_master():
            self.save_checkpoint(state_dict, checkpoint)

    def save_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: str):
        """
        Save optimizer to checkpoint but only on master process.
        """
        # TODO(ver217): optimizer state dict is sharded
        super().save_unsharded_optimizer(optimizer, checkpoint)

    def save_lr_scheduler(self, lr_scheduler: LRScheduler, checkpoint: str):
        """
        Save model to checkpoint but only on master process.
        """
        if self.coordinator.is_master():
            super().save_lr_scheduler(lr_scheduler, checkpoint)


class GeminiModel(ModelWrapper):

    def __init__(self, module: nn.Module, gemini_config: dict) -> None:
        super().__init__(module)
        # TODO: only support Gemini now
        self.module = zero_model_wrapper(module, zero_stage=3, gemini_config=gemini_config)

    def unwrap(self):
        # as save/load state dict is coupled with the GeminiDDP, we only return GeminiDDP model
        return self.module


class GeminiOptimizer(OptimizerWrapper):

    def __init__(self, module: GeminiDDP, optimizer: Optimizer, zero_optim_config: dict, optim_kwargs: dict) -> None:
        optimizer = zero_optim_wrapper(module, optimizer, optim_config=zero_optim_config, **optim_kwargs)
        super().__init__(optimizer)

    def backward(self, loss: Tensor, *args, **kwargs):
        self.optim.backward(loss)

    def clip_grad_by_norm(self,
                          max_norm: Union[float, int],
                          norm_type: Union[float, int] = 2,
                          error_if_nonfinite: bool = False,
                          *args,
                          **kwargs) -> Tensor:
        warnings.warn(f'Gemini controls grad clipping by itself, so you should not use clip_grad_by_norm')

    def clip_grad_by_value(self, clip_value: float, *args, **kwargs) -> None:
        raise NotImplementedError('Gemini does not support clip_grad_by_value')


class GeminiPlugin(Plugin):
    """
    Plugin for Gemini.

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

    def __init__(
        self,
        device: torch.device = get_current_device(),
        placement_policy: str = "cpu",
        pin_memory: bool = False,
        force_outputs_fp32: bool = False,
        strict_ddp_mode: bool = False,
        search_range_mb: int = 32,
        hidden_dim: Optional[int] = None,
        min_chunk_size_mb: float = 32,
        memstats: Optional[MemStats] = None,
        gpu_margin_mem_ratio: float = 0.0,
        initial_scale: float = 2**32,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
        max_norm: float = 0.0,
        norm_type: float = 2.0,
    ) -> None:

        assert dist.is_initialized(
        ), 'torch.distributed is not initialized, please use colossalai.launch to create the distributed environment'
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.gemini_config = dict(
            device=device,
            placement_policy=placement_policy,
            pin_memory=pin_memory,
            force_outputs_fp32=force_outputs_fp32,
            strict_ddp_mode=strict_ddp_mode,
            search_range_mb=search_range_mb,
            hidden_dim=hidden_dim,
            min_chunk_size_mb=min_chunk_size_mb,
            memstats=memstats,
        )
        self.zero_optim_config = dict(gpu_margin_mem_ratio=gpu_margin_mem_ratio,)
        self.optim_kwargs = dict(initial_scale=initial_scale,
                                 growth_factor=growth_factor,
                                 backoff_factor=backoff_factor,
                                 growth_interval=growth_interval,
                                 hysteresis=hysteresis,
                                 min_scale=min_scale,
                                 max_scale=max_scale,
                                 max_norm=max_norm,
                                 norm_type=norm_type)

    def support_no_sync(self) -> bool:
        return False

    def control_precision(self) -> bool:
        return True

    def supported_precisions(self) -> List[str]:
        return ['fp16']

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

        if not isinstance(model, ModelWrapper):
            # convert model to sync bn
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model, None)

            # wrap the model with PyTorch DDP
            model = GeminiModel(model, self.gemini_config)

        if not isinstance(optimizer, OptimizerWrapper):
            optimizer = GeminiOptimizer(model.unwrap(), optimizer, self.zero_optim_config, self.optim_kwargs)

        return model, optimizer, criterion, dataloader, lr_scheduler

    def control_checkpoint_io(self) -> bool:
        return True

    def get_checkpoint_io(self) -> CheckpointIO:
        return GeminiCheckpointIO()
