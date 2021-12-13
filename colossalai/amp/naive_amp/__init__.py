import torch.nn as nn
from torch.optim import Optimizer
from colossalai.utils import is_no_pp_or_last_stage

from .naive_amp import NaiveAMPOptimizer, NaiveAMPModel


def convert_to_naive_amp(model: nn.Module,
                         optimizer: Optimizer,
                         amp_config):
    """A helper function to wrap training components with Torch AMP modules

    :param model: your model object
    :type model: :class:`torch.nn.Module`
    :param optimizer: your optimizer object
    :type optimizer: :class:`torch.optim.Optimzer`
    :param amp_config: configuration for naive mode amp
    :type amp_config: :class:`colossalai.context.Config` or dict

    :return: (model, optimizer)
    :rtype: Tuple
    """
    if is_no_pp_or_last_stage():
        model = NaiveAMPModel(model, output_to_fp32=True)
    else:
        model = NaiveAMPModel(model, output_to_fp32=False)

    optimizer = NaiveAMPOptimizer(optimizer, **amp_config)
    return model, optimizer


__all__ = ['convert_to_naive_amp', 'NaiveAMPOptimizer']
