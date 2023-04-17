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
from colossalai.checkpoint_io.utils import save_state_dict
from colossalai.cluster import DistCoordinator
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.utils import get_current_device
from colossalai.zero import GeminiDDP, zero_model_wrapper, zero_optim_wrapper
from colossalai.zero.gemini.memory_tracer import MemStats

from .plugin_base import Plugin

__all__ = ['GeminiPlugin']


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

    def save_unsharded_model(self, model: GeminiDDP, checkpoint: str, gather_dtensor: bool, use_safetensors: bool):
        """
        Save model to checkpoint but only on master process.
        """
        # the model should be unwrapped in self.load_model via ModelWrapper.unwrap
        # as there is communication when get state dict, this must be called on all processes
        state_dict = model.state_dict(only_rank_0=True)
        if self.coordinator.is_master():
            save_state_dict(state_dict, checkpoint, use_safetensors)

    def save_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: str, gather_dtensor: bool):
        """
        Save optimizer to checkpoint but only on master process.
        """
        # TODO(ver217): optimizer state dict is sharded
        super().save_unsharded_optimizer(optimizer, checkpoint, gather_dtensor)

    def save_lr_scheduler(self, lr_scheduler: LRScheduler, checkpoint: str):
        """
        Save model to checkpoint but only on master process.
        """
        if self.coordinator.is_master():
            super().save_lr_scheduler(lr_scheduler, checkpoint)


class GeminiModel(ModelWrapper):

    def __init__(self, module: nn.Module, gemini_config: dict, verbose: bool = False) -> None:
        super().__init__(module)
        self.module = zero_model_wrapper(module, zero_stage=3, gemini_config=gemini_config, verbose=verbose)

    def unwrap(self):
        # as save/load state dict is coupled with the GeminiDDP, we only return GeminiDDP model
        return self.module


class GeminiOptimizer(OptimizerWrapper):

    def __init__(self,
                 module: GeminiDDP,
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
        warnings.warn(f'Gemini controls grad clipping by itself, so you should not use clip_grad_by_norm')

    def clip_grad_by_value(self, clip_value: float, *args, **kwargs) -> None:
        raise NotImplementedError('Gemini does not support clip_grad_by_value')


class GeminiPlugin(Plugin):
    """
    Plugin for Gemini.

    Example:
        >>> from colossalai.booster import Booster
        >>> from colossalai.booster.plugin import GeminiPlugin
        >>>
        >>> model, train_dataset, optimizer, criterion = ...
        >>> plugin = GeminiPlugin()

        >>> train_dataloader = plugin.prepare_train_dataloader(train_dataset, batch_size=8)
        >>> booster = Booster(plugin=plugin)
        >>> model, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)

    Args:
        device (torch.device): device to place the model.
        placement_policy (str, optional): "cpu", "cuda", "auto". Defaults to "cpu".
        pin_memory (bool, optional): use pin memory on CPU. Defaults to False.
        force_outputs_fp32 (bool, optional): force outputs are fp32. Defaults to False.
        strict_ddp_mode (bool, optional): use strict ddp mode (only use dp without other parallelism). Defaults to False.
        search_range_mb (int, optional): chunk size searching range in MegaByte. Defaults to 32.
        hidden_dim (int, optional): the hidden dimension of DNN.
            Users can provide this argument to speed up searching.
            If users do not know this argument before training, it is ok. We will use a default value 1024.
        min_chunk_size_mb (float, optional): the minimum chunk size in MegaByte.
            If the aggregate size of parameters is still samller than the minimum chunk size,
            all parameters will be compacted into one small chunk.
        memstats (MemStats, optional) the memory statistics collector by a runtime memory tracer.
        gpu_margin_mem_ratio (float, optional): The ratio of GPU remaining memory (after the first forward-backward)
            which will be used when using hybrid CPU optimizer.
            This argument is meaningless when `placement_policy` of `GeminiManager` is not "auto".
            Defaults to 0.0.
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
        verbose (bool, optional): verbose mode. Debug info including chunk search result will be printed. Defaults to False.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
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
        verbose: bool = False,
    ) -> None:

        assert dist.is_initialized(
        ), 'torch.distributed is not initialized, please use colossalai.launch to create the distributed environment'
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.gemini_config = dict(
            device=(device or get_current_device()),
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
        self.verbose = verbose

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
            # FIXME(ver217): gemini does not support sync bn
            # In torch/nn/modules/_functions.py, line 22, ``mean, invstd = torch.batch_norm_stats(input, eps)`` will get fp32 mean and invstd even though the input is fp16.
            # This inconsistency of dtype will cause the error.
            # We have two possible solutions:
            # 1. keep batch norm always in fp32. This is hard for gemini, as it use chunks.
            # 2. patch sync bn or write a new on. This is relatively easy, but we need to test it.
            # model = nn.SyncBatchNorm.convert_sync_batchnorm(model, None)

            # wrap the model with Gemini
            model = GeminiModel(model, self.gemini_config, self.verbose)

        if not isinstance(optimizer, OptimizerWrapper):
            optimizer = GeminiOptimizer(model.unwrap(), optimizer, self.zero_optim_config, self.optim_kwargs,
                                        self.verbose)

        return model, optimizer, criterion, dataloader, lr_scheduler

    def control_checkpoint_io(self) -> bool:
        return True

    def get_checkpoint_io(self) -> CheckpointIO:
        return GeminiCheckpointIO()
