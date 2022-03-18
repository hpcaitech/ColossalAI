from .apex_amp import ApexAMPOptimizer
import torch.nn as nn
from torch.optim import Optimizer


def convert_to_apex_amp(model: nn.Module, optimizer: Optimizer, amp_config):
    """A helper function to wrap training components with Apex AMP modules

    :param model: your model object
    :type model: :class:`torch.nn.Module`
    :param optimizer: your optimizer object
    :type optimizer: :class:`torch.optim.Optimizer`
    :param amp_config: configuration for nvidia apex
    :type amp_config: :class:`colossalai.context.Config` or dict
    :return: (model, optimizer)
    :rtype: Tuple

    amp_config should contain the parameters below
    {
    enabled:bool (optional, default to be True),
    opt_level:str (optional, default to be 'O1'),
    cast_model_type:torch.dtype (optional, default to be None),
    patch_torch_functions:bool (optional, default to be None),
    keep_batchnorm_fp32:bool or str (optional, default to be None),
    master_weights:bool (optional, default to be None),
    loss_scale:float or str (optional, default to be None),
    cast_model_outputs:torch.dpython:type (optional, default to be None),
    num_losses:int (optional, default to be 1),
    verbosity:int (optional, default to be 1),
    min_loss_scale:float (optional, default to be None),
    max_loss_scale:float (optional, default to be 2.**24)
    }

    more details about amp_config refer to https://nvidia.github.io/apex/amp.html?highlight=apex%20amp
    """
    import apex.amp as apex_amp
    model, optimizer = apex_amp.initialize(model, optimizer, **amp_config)
    optimizer = ApexAMPOptimizer(optimizer)
    return model, optimizer


__all__ = ['convert_to_apex_amp', 'ApexAMPOptimizer']
