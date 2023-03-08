from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from .plugin import Plugin

__all__ = ['Booster']


class Booster:

    def __init__(self,
                 device: Union[str, torch.device] = 'cuda',
                 precision: str = 'fp32',
                 grad_clipping_type: str = 'norm',
                 grad_clipping_value: float = 0.0,
                 plugin: Optional[Plugin] = None) -> None:
        # TODO: implement this method
        pass

    def boost(
        self, *args: Union[nn.Module, Optimizer, LRScheduler, DataLoader]
    ) -> List[Union[nn.Module, Optimizer, LRScheduler, DataLoader]]:
        # TODO: implement this method
        pass

    def backward(self, loss: torch.Tensor, optimizer: Optimizer) -> None:
        # TODO: implement this method
        pass

    def execute_pipeline(self,
                         data_iter: Iterator,
                         model: nn.Module,
                         criterion: Callable[[torch.Tensor], torch.Tensor],
                         optimizer: Optimizer,
                         return_loss: bool = True,
                         return_outputs: bool = False) -> Tuple[Optional[torch.Tensor], ...]:
        # TODO: implement this method
        # run pipeline forward backward pass
        # return loss or outputs if needed
        pass

    def no_sync(self, model: nn.Module) -> contextmanager:
        # TODO: implement this method
        pass

    def save(self,
             obj: Union[nn.Module, Optimizer, LRScheduler],
             path_like: str,
             plan: str = 'torch',
             **kwargs) -> None:
        # TODO: implement this method
        pass

    def load(self,
             obj: Union[nn.Module, Optimizer, LRScheduler],
             path_like: str,
             plan: str = 'torch',
             **kwargs) -> None:
        # TODO: implement this method
        pass
