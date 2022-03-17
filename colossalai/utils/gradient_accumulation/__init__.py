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
    """
    :param model: your model object
    :type model: :class:`torch.nn.Module`
    :param optimizer: your optimizer object
    :type optimizer: :class:`torch.optim.Optimizer`
    :param dataloader: your dataloader object, would be called like iter(dataloader)
    :type dataloader: torch.utils.data.DataLoader or iterable objects
    :param accumulate_size: the number of steps to accumulate gradients
    :type accumulate_size: int
    :param gradient_handlers: list of gradient handler objects. Default is None
    :type gradient_handlers: List[:class:`colossalai.engine.BaseGradientHandler`]
    :param lr_scheduler: your lr scheduler object. Default is None
    :type lr_scheduler: `torch.optim.lr_scheduler` or `colossalai.nn.lr_scheduler`

    more details about `gradient_handlers` could be found in .../engine/gradient_handler or
    https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/engine/gradient_handler

    more details about `lr_scheduler` could be found in .../nn/lr_scheduler or
    https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/nn/lr_scheduler and
    https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    """
    optimizer = GradAccumOptimizer(optimizer, accumulate_size=accumulate_size, model=model)
    dataloader = GradAccumDataloader(dataloader, accumulate_size=accumulate_size)

    if gradient_handlers is not None:
        gradient_handlers = [GradAccumGradientHandler(handler, accumulate_size) for handler in gradient_handlers]

    if lr_scheduler is not None:
        lr_scheduler = GradAccumLrSchedulerByStep(lr_scheduler, accumulate_size=accumulate_size)

    return optimizer, dataloader, gradient_handlers, lr_scheduler


__all__ = ['accumulate_gradient', 'GradAccumDataloader', 'GradAccumOptimizer',
           'GradAccumLrSchedulerByStep', 'GradAccumGradientHandler']
