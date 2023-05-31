import logging
import os
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from colossalai.checkpoint_io import CheckpointIO, GeneralCheckpointIO
from colossalai.checkpoint_io.utils import load_state_dict, save_state_dict
from colossalai.cluster import DistCoordinator
from colossalai.elixir import ElixirModule, ElixirOptimizer
from colossalai.elixir.cuda import set_memory_fraction
from colossalai.elixir.search import minimum_waste_search, optimal_search
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.utils import get_current_device

from .dp_plugin_base import DPPluginBase

__all__ = ['ElixirPlugin']


class ElixirCheckpointIO(GeneralCheckpointIO):

    def __init__(self) -> None:
        super().__init__()
        self.coordinator = DistCoordinator()

    def load_unsharded_model(self, model: ElixirModule, checkpoint: str):
        """
        Load available model states from checkpoint.
        """
        if self.coordinator.is_master():
            checkpoint = load_state_dict(checkpoint)
        else:
            checkpoint = None
        model.load_state_dict(checkpoint, only_rank_0=True)

    def save_unsharded_model(self, model: ElixirModule, checkpoint: str, use_safetensors: bool = False):
        """
        Save model states to checkpoint but only on master process.
        """
        state_dict = model.state_dict(only_rank_0=True)
        if self.coordinator.is_master():
            save_state_dict(state_dict, checkpoint, use_safetensors)

    def save_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: str, gather_dtensor: bool):
        """
        Save optimizer to checkpoint but only on master process.
        """
        # TODO: optimizer state dict is sharded
        warnings.warn('ElixirPlugin does not support save full optimizer checkpoint now. Save it on every process.')
        checkpoint = f'{checkpoint}.rank{self.coordinator.rank}'
        super().save_unsharded_optimizer(optimizer, checkpoint, gather_dtensor)

    def load_optimizer(self, optimizer: Optimizer, checkpoint: str):
        warnings.warn(
            'ElixirPlugin can only load optimizer checkpoint saved by itself with the same number of processes.')
        checkpoint = f'{checkpoint}.rank{self.coordinator.rank}'
        super().load_optimizer(optimizer, checkpoint)

    def save_lr_scheduler(self, lr_scheduler: LRScheduler, checkpoint: str):
        """
        Save model to checkpoint but only on master process.
        """
        if self.coordinator.is_master():
            super().save_lr_scheduler(lr_scheduler, checkpoint)


class ELXModel(ModelWrapper):

    def __init__(self, module: nn.Module, search_func: Callable, search_config: Dict, module_config: Dict) -> None:
        super().__init__(module)
        sr = search_func(module, **search_config)
        self.module = ElixirModule(module, sr, **module_config)

    def unwrap(self):
        # just return the ElixirModule instance
        return self.module


class ELXOptimizer(OptimizerWrapper):

    def __init__(self, module: ElixirModule, optimizer: Optimizer, optimizer_config: dict) -> None:
        optimizer = ElixirOptimizer(module, optimizer, **optimizer_config, init_step=True)
        super().__init__(optimizer)

    def backward(self, loss: Tensor, *args, **kwargs):
        self.optim.backward(loss)

    def clip_grad_by_norm(self,
                          max_norm: Union[float, int],
                          norm_type: Union[float, int] = 2,
                          error_if_nonfinite: bool = False,
                          *args,
                          **kwargs) -> Tensor:
        warnings.warn(f'Elixir controls grad clipping by itself, so you should set the max_norm before training.')

    def clip_grad_by_value(self, clip_value: float, *args, **kwargs) -> None:
        raise NotImplementedError('Elixir does not support clip_grad_by_value')


class ElixirPlugin(DPPluginBase):
    """
    Plugin for Elixir.

    Example:
        >>> from colossalai.booster import Booster
        >>> from colossalai.booster.plugin import ElixirPlugin
        >>>
        >>> model, train_dataset, optimizer, criterion = ...
        >>> plugin = ElixirPlugin()

        >>> train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)
        >>> booster = Booster(plugin=plugin)
        >>> model, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)

    Args:
        search_type (str): The used search algorithm for the chunk initialization, 'mini_waste' or 'optimal'.
        dtype (torch.dtype): The data type used in computations, torch.float or torch.float16.
            If torch.float16 is used, AMP is enabled automatically.
        prefetch (bool): Whether to prefetch chunks for overlapping.
            Users should provide example_input and example_step_fn if prefetch is True.
        cpu_offload (bool): Whether to offload optimizer states (OS).
            Only available when the search_type is 'mini_waste'.
        pin_memory (bool): Whether to store OS in the pinned cpu memory.
            Only available when cpu_offload is enabled.
        reduce_always_fp32 (bool): Whether to reduce gradients in fp32.
        outputs_always_fp32 (bool): Whether to cast outputs to fp32.
        example_input (Dict): An example input for the model.
        example_step_fn (Callable): A callable function that takes the model and the example input as input, and does a training step.
        optimizer_type (str): The type of optimizer, 'Adam' or 'SGD'.
            Only used when the search type is 'optimal'.
        optimizer_config (Dict): The config of the optimizer.
            This config is commonly used in AMP.
            See the class `ElixirOptimizer` for more details.
        cuda_memory_fraction (float): The fraction of the GPU memory used Elixir.
    """

    def __init__(self,
                 search_type: str = 'mini_waste',
                 dtype: torch.dtype = torch.float32,
                 prefetch: bool = False,
                 cpu_offload: bool = False,
                 pin_memory: bool = False,
                 reduce_always_fp32: bool = False,
                 outputs_always_fp32: bool = False,
                 example_input: Optional[Dict] = None,
                 example_step_fn: Optional[Callable] = None,
                 optimizer_type: str = 'Adam',
                 optimizer_config: Optional[Dict] = None,
                 cuda_memory_fraction: float = 1.0,
                 verbose: bool = False) -> None:
        super().__init__()
        assert search_type in {'mini_waste', 'optimal'}
        assert dtype in {torch.float, torch.float16}
        self.dtype = dtype
        self.verbose = verbose
        self.world_size = dist.get_world_size()
        self.world_group = dist.group.WORLD
        set_memory_fraction(fraction=cuda_memory_fraction)

        if search_type == 'mini_waste':
            self.search_func = minimum_waste_search
            self.search_config = dict(group_size=self.world_size,
                                      unified_dtype=self.dtype,
                                      prefetch=prefetch,
                                      cpu_offload=cpu_offload,
                                      pin_memory=pin_memory,
                                      inp=example_input,
                                      step_fn=example_step_fn,
                                      verbose=self.verbose)
        elif search_type == 'optimal':
            self.search = optimal_search
            self.search_config = dict(group_size=self.world_size,
                                      unified_dtype=self.dtype,
                                      optimizer_type=optimizer_type,
                                      overlap=prefetch,
                                      inp=example_input,
                                      step_fn=example_step_fn,
                                      verbose=self.verbose)
        else:
            raise NotImplementedError

        self.module_config = dict(process_group=self.world_group,
                                  prefetch=prefetch,
                                  dtype=self.dtype,
                                  reduce_always_fp32=reduce_always_fp32,
                                  output_fp32=outputs_always_fp32)

        if optimizer_config is None:
            optimizer_config = dict()
        self.optimizer_config = optimizer_config

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

    def configure(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Callable = None,
        dataloader: DataLoader = None,
        lr_scheduler: LRScheduler = None,
    ) -> Tuple[Union[nn.Module, OptimizerWrapper, LRScheduler, DataLoader]]:

        if not isinstance(model, ModelWrapper):
            model = ELXModel(module=model,
                             search_func=self.search_func,
                             search_config=self.search_config,
                             module_config=self.module_config)

        if not isinstance(optimizer, OptimizerWrapper):
            optimizer = ELXOptimizer(module=model.unwrap(), optimizer=optimizer, optimizer_config=self.optimizer_config)

        return model, optimizer, criterion, dataloader, lr_scheduler

    def control_checkpoint_io(self) -> bool:
        return True

    def get_checkpoint_io(self) -> CheckpointIO:
        return ElixirCheckpointIO()

    def no_sync(self, model: nn.Module) -> Iterator[None]:
        raise NotImplementedError
