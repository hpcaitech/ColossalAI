import inspect

import torch.nn as nn
from torch.optim import Optimizer

from colossalai.amp.naive_amp.grad_scaler import ConstantGradScaler, DynamicGradScaler
from colossalai.legacy.utils import is_no_pp_or_last_stage

from ._fp16_optimizer import FP16Optimizer
from .naive_amp import NaiveAMPModel, NaiveAMPOptimizer


def convert_to_naive_amp(model: nn.Module, optimizer: Optimizer, amp_config):
    """A helper function to wrap training components with naive AMP modules. In this mode,
    we forcibly cast the model weights and inputs to FP16, and cast the model outputs to FP32 to calculate loss,
    which is equivalent to Apex O3.

    Args:
        model (:class:`torch.nn.Module`): your model object
        optimizer (:class:`torch.optim.Optimizer`): your optimizer object
        amp_config (:class:`colossalai.context.Config` or dict): configuration for naive mode amp.

    Returns:
        Tuple: A tuple (model, optimizer)

    The ``amp_config`` should contain parameters below::

        verbose (bool, optional): if set to `True`, will print debug info (Default: False).
        clip_grad_norm (float, optional): clip gradients with this global L2 norm (Default 0).
                                          Note that clipping is ignored if clip_grad == 0.
        dynamic_grad_scale (bool): whether to use dynamic grad scaler.
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

    use_dynamic_grad_scaler = amp_config.pop("dynamic_grad_scale", True)
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


__all__ = ["convert_to_naive_amp", "NaiveAMPOptimizer", "FP16Optimizer"]
