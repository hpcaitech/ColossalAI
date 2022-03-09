import torch.nn as nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from colossalai.context import Config
from .torch_amp import TorchAMPOptimizer, TorchAMPModel, TorchAMPLoss
from typing import Optional


def convert_to_torch_amp(model: nn.Module,
                         optimizer: Optimizer,
                         criterion: Optional[_Loss] = None,
                         amp_config: Optional[Config] = None):
    """A helper function to wrap training components with Torch AMP modules

    :param model: your model object
    :type model: :class:`torch.nn.Module`
    :param optimizer: your optimizer object
    :type optimizer: :class:`torch.optim.Optimzer`
    :param criterion: your loss function object
    :type criterion: :class:`torch.nn.modules.loss._Loss`, optional
    :param amp_config: configuration for different amp modes
    :type amp_config: :class:`colossalai.context.Config` or dict, optional
    :return: (model, optimizer, criterion)
    :rtype: Tuple
    """
    model = TorchAMPModel(model)
    if amp_config is None:
        amp_config = dict()
    optimizer = TorchAMPOptimizer(optimizer, **amp_config)
    if criterion:
        criterion = TorchAMPLoss(criterion)
    return model, optimizer, criterion


__all__ = ['convert_to_torch_amp', 'TorchAMPModel', 'TorchAMPLoss', 'TorchAMPOptimizer']
