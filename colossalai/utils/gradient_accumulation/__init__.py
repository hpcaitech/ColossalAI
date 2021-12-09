import torch.nn as nn
from typing import List
from colossalai.engine import BaseGradientHandler
from typing import Iterable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from ._gradient_accumulation import GradAccumDataloader, GradAccumOptimizer, GradAccumLrSchedulerByStep, GradAccumGradientHandler


def accumulate_gradient(model: nn.Module,
                        optimizer: Optimizer,
                        dataloader: Iterable,
                        accumulate_size: int,
                        gradient_handlers: List[BaseGradientHandler] = None,
                        lr_scheduler: _LRScheduler = None):
    optimizer = GradAccumOptimizer(optimizer, accumulate_size=accumulate_size, model=model)
    dataloader = GradAccumDataloader(dataloader, accumulate_size=accumulate_size)

    if gradient_handlers is not None:
        gradient_handlers = [GradAccumGradientHandler(handler, accumulate_size) for handler in gradient_handlers]

    if lr_scheduler is not None:
        lr_scheduler = GradAccumLrSchedulerByStep(lr_scheduler, accumulate_size=accumulate_size)

    return optimizer, dataloader, gradient_handlers, lr_scheduler


__all__ = ['accumulate_gradient', 'GradAccumDataloader', 'GradAccumOptimizer',
           'GradAccumLrSchedulerByStep', 'GradAccumGradientHandler']
