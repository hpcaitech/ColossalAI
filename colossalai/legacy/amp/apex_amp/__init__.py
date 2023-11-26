import torch.nn as nn
from torch.optim import Optimizer

from .apex_amp import ApexAMPOptimizer


def convert_to_apex_amp(model: nn.Module, optimizer: Optimizer, amp_config):
    r"""A helper function to wrap training components with Apex AMP modules

    Args:
        model (:class:`torch.nn.Module`): your model object.
        optimizer (:class:`torch.optim.Optimizer`): your optimizer object.
        amp_config (Union[:class:`colossalai.context.Config`, dict]): configuration for initializing apex_amp.

    Returns:
        Tuple: A tuple (model, optimizer).

    The ``amp_config`` should include parameters below:
    ::

        enabled (bool, optional, default=True)
        opt_level (str, optional, default="O1")
        cast_model_type (``torch.dtype``, optional, default=None)
        patch_torch_functions (bool, optional, default=None)
        keep_batchnorm_fp32 (bool or str, optional, default=None
        master_weights (bool, optional, default=None)
        loss_scale (float or str, optional, default=None)
        cast_model_outputs (torch.dtype, optional, default=None)
        num_losses (int, optional, default=1)
        verbosity (int, default=1)
        min_loss_scale (float, default=None)
        max_loss_scale (float, default=2.**24)

    More details about ``amp_config`` refer to `amp_config <https://nvidia.github.io/apex/amp.html?highlight=apex%20amp>`_.
    """
    import apex.amp as apex_amp

    model, optimizer = apex_amp.initialize(model, optimizer, **amp_config)
    optimizer = ApexAMPOptimizer(optimizer)
    return model, optimizer


__all__ = ["convert_to_apex_amp", "ApexAMPOptimizer"]
