from typing import Callable, Iterator, List, Optional, Tuple, Union

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from colossalai.checkpoint_io import CheckpointIO, GeneralCheckpointIO
from colossalai.cluster import DistCoordinator
from colossalai.interface import ModelWrapper, OptimizerWrapper

from .dp_plugin_base import DPPluginBase

__all__ = ['TorchDDPPlugin']


class TorchDDPCheckpointIO(GeneralCheckpointIO):

    def __init__(self) -> None:
        super().__init__()
        self.coordinator = DistCoordinator()

    def load_unsharded_model(self, model: nn.Module, checkpoint: str, strict: bool = True):
        """
        Load model from checkpoint with automatic unwrapping.
        """
        # the model should be unwrapped in self.load_model via ModelWrapper.unwrap
        return super().load_unsharded_model(model, checkpoint, strict=strict)

    def save_unsharded_model(self, model: nn.Module, checkpoint: str, gather_dtensor: bool, use_safetensors: bool):
        """
        Save model to checkpoint but only on master process.
        """
        if self.coordinator.is_master():
            super().save_unsharded_model(model, checkpoint, gather_dtensor, use_safetensors)

    def save_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: str, gather_dtensor: bool):
        """
        Save optimizer to checkpoint but only on master process.
        """
        if self.coordinator.is_master():
            super().save_unsharded_optimizer(optimizer, checkpoint, gather_dtensor)

    def save_lr_scheduler(self, lr_scheduler: LRScheduler, checkpoint: str):
        """
        Save model to checkpoint but only on master process.
        """
        if self.coordinator.is_master():
            super().save_lr_scheduler(lr_scheduler, checkpoint)

    def save_sharded_model(self,
                           model: nn.Module,
                           checkpoint_path: str,
                           gather_dtensor: bool = True,
                           prefix: Optional[str] = None,
                           max_shard_size: int = 1024,
                           use_safetensors: bool = False):
        """
        Save model to checkpoint but only on master process.
        """
        if self.coordinator.is_master():
            super().save_sharded_model(model, checkpoint_path, gather_dtensor, prefix, max_shard_size, use_safetensors)

    def save_sharded_optimizer(self,
                               optimizer: Optimizer,
                               checkpoint: str,
                               gather_dtensor: bool = True,
                               prefix: Optional[str] = None,
                               size_per_shard: int = 1024):
        """
        Save optimizer to checkpoint but only on master process.
        """
        if self.coordinator.is_master():
            super().save_sharded_optimizer(optimizer, checkpoint, gather_dtensor, prefix, size_per_shard)


class TorchDDPModel(ModelWrapper):

    def __init__(self, module: nn.Module, *args, **kwargs) -> None:
        super().__init__(module)
        self.module = DDP(module, *args, **kwargs)

    def unwrap(self):
        return self.module.module


class TorchDDPPlugin(DPPluginBase):
    """
    Plugin for PyTorch DDP.

    Example:
        >>> from colossalai.booster import Booster
        >>> from colossalai.booster.plugin import TorchDDPPlugin
        >>>
        >>> model, train_dataset, optimizer, criterion = ...
        >>> plugin = TorchDDPPlugin()

        >>> train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)
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
        super().__init__()
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

    def configure(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> Tuple[nn.Module, OptimizerWrapper, Callable, DataLoader, LRScheduler]:
        # cast model to cuda
        model = model.cuda()

        # convert model to sync bn
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model, None)

        # wrap the model with PyTorch DDP
        model = TorchDDPModel(model, **self.ddp_kwargs)

        if optimizer is not None and \
                not isinstance(optimizer, OptimizerWrapper):
            optimizer = OptimizerWrapper(optimizer)

        return model, optimizer, criterion, dataloader, lr_scheduler

    def control_checkpoint_io(self) -> bool:
        return True

    def get_checkpoint_io(self) -> CheckpointIO:
        return TorchDDPCheckpointIO()

    def no_sync(self, model: nn.Module) -> Iterator[None]:
        assert isinstance(model, TorchDDPModel), 'Model is not boosted by TorchDDPPlugin.'
        return model.module.no_sync()
