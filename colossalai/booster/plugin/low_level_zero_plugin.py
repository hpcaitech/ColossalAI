import warnings
from functools import partial
from typing import Callable, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader

from colossalai.checkpoint_io import CheckpointIO, GeneralCheckpointIO
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.utils import get_current_device
from colossalai.zero import zero_model_wrapper, zero_optim_wrapper

from .dp_plugin_base import DPPluginBase
from .torch_ddp_plugin import TorchDDPCheckpointIO

__all__ = ['LowLevelZeroPlugin']


def _convert_floating_point(x, dtype: torch.dtype = torch.float16):
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype)
    return x


SUPPORTED_PRECISION = ['fp16', 'bf16', 'fp32']


class LowLevelZeroCheckpointIO(TorchDDPCheckpointIO):

    def save_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: str, gather_dtensor: bool):
        """
        Save optimizer to checkpoint but only on master process.
        """
        # TODO(ver217): optimizer state dict is sharded, and cannot get full state dict now
        warnings.warn(
            'LowLevelZeroPlugin does not support save full optimizer checkpoint now. Save it on every process.')
        checkpoint = f'{checkpoint}.rank{self.coordinator.rank}'
        GeneralCheckpointIO.save_unsharded_optimizer(self, optimizer, checkpoint, gather_dtensor)

    def load_optimizer(self, optimizer: Optimizer, checkpoint: str):
        warnings.warn(
            'LowLevelZeroPlugin can only load optimizer checkpoint saved by itself with the same number of processes.')
        checkpoint = f'{checkpoint}.rank{self.coordinator.rank}'
        super().load_optimizer(optimizer, checkpoint)


class LowLevelZeroModel(ModelWrapper):

    def __init__(self, module: nn.Module, stage: int, precision: str) -> None:
        super().__init__(module)
        self.dtype = None
        if precision == 'fp16':
            self.dtype = torch.float16
        elif precision == 'bf16':
            self.dtype = torch.bfloat16
        module = zero_model_wrapper(module, zero_stage=stage)
        if self.dtype is not None:
            module = module.to(self.dtype)
        module = module.to(get_current_device())
        self.module = module
        self.convert_fn = None
        if self.dtype is not None:
            self.convert_fn = partial(_convert_floating_point, dtype=self.dtype)

    def forward(self, *args, **kwargs):
        if self.convert_fn is not None:
            args = tree_map(self.convert_fn, args)
            kwargs = tree_map(self.convert_fn, kwargs)
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


class LowLevelZeroPlugin(DPPluginBase):
    """
    Plugin for low level zero.

    Example:
        >>> from colossalai.booster import Booster
        >>> from colossalai.booster.plugin import LowLevelZeroPlugin
        >>>
        >>> model, train_dataset, optimizer, criterion = ...
        >>> plugin = LowLevelZeroPlugin()

        >>> train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)
        >>> booster = Booster(plugin=plugin)
        >>> model, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)

    Args:
        strage (int, optional): ZeRO stage. Defaults to 1.
        precision (str, optional): precision. Support 'fp16', 'bf16' and 'fp32'. Defaults to 'fp16'.
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
        super().__init__()
        assert stage in (1, 2), f'LowLevelZeroPlugin only supports stage 1/2 training'
        assert precision in SUPPORTED_PRECISION, f'LowLevelZeroPlugin only supports {SUPPORTED_PRECISION} training'

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
        return SUPPORTED_PRECISION

    def control_device(self) -> bool:
        return True

    def supported_devices(self) -> List[str]:
        return ['cuda']

    def configure(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> Tuple[nn.Module, OptimizerWrapper, Callable, DataLoader, LRScheduler]:

        if not isinstance(model, ModelWrapper):
            model = LowLevelZeroModel(model, self.stage, self.precision)

        if optimizer is not None and \
                not isinstance(optimizer, OptimizerWrapper):
            optimizer = LowLevelZeroOptimizer(model.unwrap(),
                                              optimizer,
                                              self.zero_optim_config,
                                              self.optim_kwargs,
                                              self.verbose)

        return model, optimizer, criterion, dataloader, lr_scheduler

    def control_checkpoint_io(self) -> bool:
        return True

    def get_checkpoint_io(self) -> CheckpointIO:
        return LowLevelZeroCheckpointIO()

    def no_sync(self, model: nn.Module) -> Iterator[None]:
        raise NotImplementedError
