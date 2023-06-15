from abc import ABC, abstractmethod
from typing import Callable, Iterator, List, Optional, Tuple, Union

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader, Dataset

from colossalai.checkpoint_io import CheckpointIO
from colossalai.interface import OptimizerWrapper

__all__ = ['Plugin']


class Plugin(ABC):

    @abstractmethod
    def supported_devices(self) -> List[str]:
        pass

    @abstractmethod
    def supported_precisions(self) -> List[str]:
        pass

    @abstractmethod
    def control_precision(self) -> bool:
        pass

    @abstractmethod
    def control_device(self) -> bool:
        pass

    @abstractmethod
    def support_no_sync(self) -> bool:
        pass

    @abstractmethod
    def configure(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> Tuple[nn.Module, OptimizerWrapper, Callable, DataLoader, LRScheduler]:
        # implement this method
        pass

    @abstractmethod
    def control_checkpoint_io(self) -> bool:
        """
        Whether the plugin controls the checkpoint io
        """
        pass

    @abstractmethod
    def get_checkpoint_io(self) -> CheckpointIO:
        """
        Get checkpoint io object for this plugin, only invoked when control_checkpoint_io is True.
        """
        pass

    @abstractmethod
    def no_sync(self, model: nn.Module) -> Iterator[None]:
        """
        Context manager to disable gradient synchronization.
        """
        pass

    @abstractmethod
    def prepare_dataloader(self,
                           dataset: Dataset,
                           batch_size: int,
                           shuffle: bool = False,
                           seed: int = 1024,
                           drop_last: bool = False,
                           pin_memory: bool = False,
                           num_workers: int = 0,
                           **kwargs):
        """Prepare a dataloader for distributed training. The dataloader will be wrapped by
        `torch.utils.data.DataLoader`
        """
        pass
