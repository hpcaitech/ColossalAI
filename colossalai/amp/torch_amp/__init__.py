import torch.nn as nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from colossalai.context import Config
from .torch_amp import TorchAMPOptimizer, TorchAMPModel, TorchAMPLoss


def convert_to_torch_amp(model: nn.Module,
                         optimizer: Optimizer,
                         criterion: _Loss,
                         amp_config: Config):
    """A helper function to wrap training components with Torch AMP modules

    :param model: your model object
    :type model: :class:`torch.nn.Module`
    :param optimizer: your optimizer object
    :type optimizer: :class:`torch.optim.Optimzer`
    :param criterion: your loss function object
    :type criterion: :class:`torch.nn.modules.loss._Loss`
    :param amp_config: configuration for different amp modes
    :type amp_config: :class:`colossalai.context.Config` or dict
    
    :return: (model, optimizer, criterion)
    :rtype: Tuple
    """
    model = TorchAMPModel(model)
    optimizer = TorchAMPOptimizer(optimizer, **amp_config)
    criterion = TorchAMPLoss(criterion)
    return model, optimizer, criterion


__all__ = ['convert_to_torch_amp', 'TorchAMPModel', 'TorchAMPLoss', 'TorchAMPOptimizer']
