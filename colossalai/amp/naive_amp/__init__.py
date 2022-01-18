import torch.nn as nn
from torch.optim import Optimizer
from colossalai.utils import is_no_pp_or_last_stage

from .naive_amp import NaiveAMPOptimizer, NaiveAMPModel


def convert_to_naive_amp(model: nn.Module,
                         optimizer: Optimizer,
                         amp_config):
    """A helper function to wrap training components with naive AMP modules

    :param model: your model object
    :type model: :class:`torch.nn.Module`
    :param optimizer: your optimizer object
    :type optimizer: :class:`torch.optim.Optimzer`
    :param amp_config: configuration for naive mode amp
    :type amp_config: :class:`colossalai.context.Config` or dict

    :return: (model, optimizer)
    :rtype: Tuple
    """
    if isinstance(model, nn.ModuleList):
        # interleaved pipeline
        module_list = []
        for chunk, m in enumerate(model):
            output_to_fp32 = is_no_pp_or_last_stage() and chunk == len(model) - 1
            module_list.append(NaiveAMPModel(m, output_to_fp32=output_to_fp32))
        model = nn.ModuleList(module_list)
    else:
        output_to_fp32 = is_no_pp_or_last_stage()
        model = NaiveAMPModel(model, output_to_fp32=output_to_fp32)

    optimizer = NaiveAMPOptimizer(optimizer, **amp_config)
    return model, optimizer


__all__ = ['convert_to_naive_amp', 'NaiveAMPOptimizer']
