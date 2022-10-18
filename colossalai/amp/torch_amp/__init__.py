from typing import Optional

import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from colossalai.context import Config

from .torch_amp import TorchAMPLoss, TorchAMPModel, TorchAMPOptimizer


def convert_to_torch_amp(model: nn.Module,
                         optimizer: Optimizer,
                         criterion: Optional[_Loss] = None,
                         amp_config: Optional[Config] = None):
    """A helper function to wrap training components with Pytorch AMP modules

    Args:
        model (:class:`torch.nn.Module`): your model object.
        optimizer (:class:`torch.optim.Optimizer`): your optimizer object
        criterion (:class:`torch.nn.modules.loss._Loss`, optional): your loss function object
        amp_config (:class:`colossalai.context.Config` or dict, optional): configuration for Pytorch AMP.

    The ``amp_config`` should include parameters below:
    ::

        init_scale (float, optional, default=2.**16)
        growth_factor (float, optional, default=2.0)
        backoff_factor (float, optional, default=0.5)
        growth_interval (int, optional, default=2000)
        enabled (bool, optional, default=True)

    Returns:
        A tuple (model, optimizer, criterion)
    """
    model = TorchAMPModel(model)
    if amp_config is None:
        amp_config = dict()
    optimizer = TorchAMPOptimizer(optimizer, **amp_config)
    if criterion:
        criterion = TorchAMPLoss(criterion)
    return model, optimizer, criterion


__all__ = ['convert_to_torch_amp', 'TorchAMPModel', 'TorchAMPLoss', 'TorchAMPOptimizer']
