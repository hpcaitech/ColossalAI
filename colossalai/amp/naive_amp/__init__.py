import inspect
import torch.nn as nn
from torch.optim import Optimizer
from colossalai.utils import is_no_pp_or_last_stage
from .naive_amp import NaiveAMPOptimizer, NaiveAMPModel
from .grad_scaler import DynamicGradScaler, ConstantGradScaler


def convert_to_naive_amp(model: nn.Module, optimizer: Optimizer, amp_config):
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

    use_dynamic_grad_scaler = amp_config.pop('dynamic_grad_scale', True)
    if use_dynamic_grad_scaler:
        scaler_class = DynamicGradScaler
    else:
        scaler_class = ConstantGradScaler

    sig = inspect.signature(scaler_class.__init__)
    kwargs = dict()
    for param in sig.parameters.values():
        if param.name in amp_config:
            kwargs[param.name] = amp_config.pop(param.name)
    grad_scaler = scaler_class(**kwargs)
    optimizer = NaiveAMPOptimizer(optimizer, grad_scaler, **amp_config)
    return model, optimizer


__all__ = ['convert_to_naive_amp', 'NaiveAMPOptimizer']
