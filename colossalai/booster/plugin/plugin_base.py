from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Union

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

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
        optimizer: Optimizer,
        criterion: Callable = None,
        dataloader: DataLoader = None,
        lr_scheduler: LRScheduler = None,
    ) -> Tuple[Union[nn.Module, OptimizerWrapper, LRScheduler, DataLoader]]:
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
