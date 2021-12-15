from .apex_amp import ApexAMPOptimizer
import torch.nn as nn
from torch.optim import Optimizer
import apex.amp as apex_amp


def convert_to_apex_amp(model: nn.Module,
                        optimizer: Optimizer,
                        amp_config):
    """A helper function to wrap training components with Torch AMP modules

    :param model: your model object
    :type model: :class:`torch.nn.Module`
    :param optimizer: your optimizer object
    :type optimizer: :class:`torch.optim.Optimzer`
    :param amp_config: configuration for nvidia apex
    :type amp_config: :class:`colossalai.context.Config` or dict

    :return: (model, optimizer)
    :rtype: Tuple
    """
    model, optimizer = apex_amp.initialize(model, optimizer, **amp_config)
    optimizer = ApexAMPOptimizer(optimizer)
    return model, optimizer


__all__ = ['convert_to_apex_amp', 'ApexAMPOptimizer']
