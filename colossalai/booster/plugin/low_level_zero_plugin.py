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
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from colossalai.checkpoint_io import CheckpointIO
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.utils import get_current_device
from colossalai.zero import zero_model_wrapper, zero_optim_wrapper

from .plugin_base import Plugin
from .torch_ddp_plugin import TorchDDPCheckpointIO

__all__ = ['LowLevelZeroPlugin']


def _convert_to_fp16(x):
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.half()
    return x


class LowLevelZeroCheckpointIO(TorchDDPCheckpointIO):

    def save_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: str, gather_dtensor: bool):
        """
        Save optimizer to checkpoint but only on master process.
        """
        # TODO(ver217): optimizer state dict is sharded
        super().save_unsharded_optimizer(optimizer, checkpoint, gather_dtensor)


class LowLevelZeroModel(ModelWrapper):

    def __init__(self, module: nn.Module, stage: int, precision: str) -> None:
        super().__init__(module)
        self.convert_inputs = (precision == 'fp16')
        module = zero_model_wrapper(module, zero_stage=stage)
        if precision == 'fp16':
            module = module.half()
        module = module.to(get_current_device())
        self.module = module

    def forward(self, *args, **kwargs):
        if self.convert_inputs:
            args = tree_map(_convert_to_fp16, args)
            kwargs = tree_map(_convert_to_fp16, kwargs)
        return super().forward(*args, **kwargs)


class LowLevelZeroOptimizer(OptimizerWrapper):

    def __init__(self,
                 module: nn.Module,
                 optimizer: Optimizer,
                 zero_optim_config: dict,
                 optim_kwargs: dict,
                 verbose: bool = False) -> None:
        optimizer = zero_optim_wrapper(module,
                                       optimizer,
                                       optim_config=zero_optim_config,
                                       **optim_kwargs,
                                       verbose=verbose)
        super().__init__(optimizer)

    def backward(self, loss: Tensor, *args, **kwargs):
        self.optim.backward(loss)

    def clip_grad_by_norm(self,
                          max_norm: Union[float, int],
                          norm_type: Union[float, int] = 2,
                          error_if_nonfinite: bool = False,
                          *args,
                          **kwargs) -> Tensor:
        warnings.warn(f'LowLevelZero controls grad clipping by itself, so you should not use clip_grad_by_norm')

    def clip_grad_by_value(self, clip_value: float, *args, **kwargs) -> None:
        raise NotImplementedError('LowLevelZero does not support clip_grad_by_value')


class LowLevelZeroPlugin(Plugin):
    """
    Plugin for low level zero.

    Example:
        >>> from colossalai.booster import Booster
        >>> from colossalai.booster.plugin import LowLevelZeroPlugin
        >>>
        >>> model, train_dataset, optimizer, criterion = ...
        >>> plugin = LowLevelZeroPlugin()

        >>> train_dataloader = plugin.prepare_train_dataloader(train_dataset, batch_size=8)
        >>> booster = Booster(plugin=plugin)
        >>> model, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)

    Args:
        strage (int, optional): ZeRO stage. Defaults to 1.
        precision (str, optional): precision. Support 'fp16' and 'fp32'. Defaults to 'fp16'.
        initial_scale (float, optional): Initial scale used by DynamicGradScaler. Defaults to 2**32.
        min_scale (float, optional): Min scale used by DynamicGradScaler. Defaults to 1.
        growth_factor (float, optional): growth_factor used by DynamicGradScaler. Defaults to 2.
        backoff_factor (float, optional): backoff_factor used by DynamicGradScaler. Defaults to 0.5.
        growth_interval (float, optional): growth_interval used by DynamicGradScaler. Defaults to 1000.
        hysteresis (float, optional): hysteresis used by DynamicGradScaler. Defaults to 2.
        max_scale (int, optional): max_scale used by DynamicGradScaler. Defaults to 2**32.
        max_norm (float, optional): max_norm used for `clip_grad_norm`. You should notice that you shall not do
            clip_grad_norm by yourself when using ZeRO DDP. The ZeRO optimizer will take care of clip_grad_norm.
        norm_type (float, optional): norm_type used for `clip_grad_norm`.
        reduce_bucket_size_in_m (int, optional): grad reduce bucket size in M. Defaults to 12.
        communication_dtype (torch.dtype, optional): communication dtype. If not specified, the dtype of param will be used. Defaults to None.
        overlap_communication (bool, optional): whether to overlap communication and computation. Defaults to True.
        cpu_offload (bool, optional): whether to offload grad, master weight and optimizer state to cpu. Defaults to False.
        verbose (bool, optional): verbose mode. Debug info including grad overflow will be printed. Defaults to False.
    """

    def __init__(
        self,
        stage: int = 1,
        precision: str = 'fp16',
        initial_scale: float = 2**32,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
        max_norm: float = 0.0,
        norm_type: float = 2.0,
        reduce_bucket_size_in_m: int = 12,
        communication_dtype: Optional[torch.dtype] = None,
        overlap_communication: bool = True,
        cpu_offload: bool = False,
        verbose: bool = False,
    ) -> None:

        assert dist.is_initialized(
        ), 'torch.distributed is not initialized, please use colossalai.launch to create the distributed environment'
        assert stage in (1, 2), f'LowLevelZeroPlugin only supports stage 1/2 training'
        assert precision in ('fp16', 'fp32'), f'LowLevelZeroPlugin only supports fp16/fp32 training'

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.stage = stage
        self.precision = precision
        self.zero_optim_config = dict(reduce_bucket_size=reduce_bucket_size_in_m * 1024 * 1024,
                                      communication_dtype=communication_dtype,
                                      overlap_communication=overlap_communication,
                                      cpu_offload=cpu_offload)
        self.optim_kwargs = dict(initial_scale=initial_scale,
                                 growth_factor=growth_factor,
                                 backoff_factor=backoff_factor,
                                 growth_interval=growth_interval,
                                 hysteresis=hysteresis,
                                 min_scale=min_scale,
                                 max_scale=max_scale,
                                 max_norm=max_norm,
                                 norm_type=norm_type)
        self.verbose = verbose

    def support_no_sync(self) -> bool:
        return False

    def control_precision(self) -> bool:
        return True

    def supported_precisions(self) -> List[str]:
        return ['fp16', 'fp32']

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
            model = LowLevelZeroModel(model, self.stage, self.precision)

        if not isinstance(optimizer, OptimizerWrapper):
            optimizer = LowLevelZeroOptimizer(model.unwrap(), optimizer, self.zero_optim_config, self.optim_kwargs,
                                              self.verbose)

        return model, optimizer, criterion, dataloader, lr_scheduler

    def control_checkpoint_io(self) -> bool:
        return True

    def get_checkpoint_io(self) -> CheckpointIO:
        return LowLevelZeroCheckpointIO()
