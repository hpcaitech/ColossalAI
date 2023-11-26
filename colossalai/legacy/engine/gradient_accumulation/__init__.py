from typing import Iterable, List

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from colossalai.legacy.engine import BaseGradientHandler

from ._gradient_accumulation import (
    GradAccumDataloader,
    GradAccumGradientHandler,
    GradAccumLrSchedulerByStep,
    GradAccumOptimizer,
)

__all__ = [
    "accumulate_gradient",
    "GradAccumDataloader",
    "GradAccumOptimizer",
    "GradAccumLrSchedulerByStep",
    "GradAccumGradientHandler",
]


def accumulate_gradient(
    model: nn.Module,
    optimizer: Optimizer,
    dataloader: Iterable,
    accumulate_size: int,
    gradient_handlers: List[BaseGradientHandler] = None,
    lr_scheduler: _LRScheduler = None,
):
    r"""Turning model, optimizer, dataloader into corresponding object for gradient accumulation.

    Args:
        model (:class:`torch.nn.Module`): your model object for gradient accumulation.
        optimizer (:class:`torch.optim.Optimizer`): your optimizer object for gradient accumulation.
        dataloader (:class:`torch.utils.data.DataLoader` or iterable objects):
            your dataloader object, would be called like iter(dataloader)
        accumulate_size (int): the number of steps to accumulate gradients
        gradient_handlers (List[:class:`colossalai.legacy.engine.BaseGradientHandler`]):
            list of gradient handler objects. Default is None.
        lr_scheduler (`torch.optim.lr_scheduler` or `colossalai.nn.lr_scheduler`):
            your ``lr_scheduler`` object for gradient accumulation. Defaults to None.

    More details about `gradient_handlers` could be found in
    `Gradient_handler <https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/engine/gradient_handler>`_.

    More details about `lr_scheduler` could be found
    `lr_scheduler <https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/nn/lr_scheduler>`_. and
    `how to adjust learning rate <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.
    """
    optimizer = GradAccumOptimizer(optimizer, accumulate_size=accumulate_size, model=model)
    dataloader = GradAccumDataloader(dataloader, accumulate_size=accumulate_size)

    if gradient_handlers is not None:
        gradient_handlers = [GradAccumGradientHandler(handler, accumulate_size) for handler in gradient_handlers]

    if lr_scheduler is not None:
        lr_scheduler = GradAccumLrSchedulerByStep(lr_scheduler, accumulate_size=accumulate_size)

    return optimizer, dataloader, gradient_handlers, lr_scheduler
