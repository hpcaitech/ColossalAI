from .apex_amp import ApexAMPOptimizer
import torch.nn as nn
from torch.optim import Optimizer


def convert_to_apex_amp(model: nn.Module, optimizer: Optimizer, amp_config):
    r"""A helper function to wrap training components with Apex AMP modules

    Args:
        model (:class:`torch.nn.Module`): your model object.
        optimizer (:class:`torch.optim.Optimizer`): your optimizer object.
        amp_config (:class:`colossalai.context.Config` or dict): configuration for nvidia apex.

    The `amp_config` should contain:

    Args:
        enabled (bool, optional, default=True):  If False, renders all Amp calls no-ops, so your script
            should run as if Amp were not present.
        opt_level (str, optional, default="O1"):  Pure or mixed precision optimization level.  Accepted values are
            "O0", "O1", "O2", and "O3", explained in detail above.
        cast_model_type (``torch.dtype``, optional, default=None):  Optional property override, see
            above.
        patch_torch_functions (bool, optional, default=None):  Optional property override.
        keep_batchnorm_fp32 (bool or str, optional, default=None):  Optional property override.  If
            passed as a string, must be the string "True" or "False".
        master_weights (bool, optional, default=None):  Optional property override.
        loss_scale (float or str, optional, default=None):  Optional property override.  If passed as a string,
            must be a string representing a number, e.g., "128.0", or the string "dynamic".
        cast_model_outputs (torch.dtype, optional, default=None):  Option to ensure that the outputs
            of your model(s) are always cast to a particular type regardless of ``opt_level``.
        num_losses (int, optional, default=1):  Option to tell Amp in advance how many losses/backward
            passes you plan to use.  When used in conjunction with the ``loss_id`` argument to
            ``amp.scale_loss``, enables Amp to use a different loss scale per loss/backward pass,
            which can improve stability.  See "Multiple models/optimizers/losses"
            under `Advanced Amp Usage`_ for examples.  If ``num_losses`` is left to 1, Amp will still
            support multiple losses/backward passes, but use a single global loss scale
            for all of them.
        verbosity (int, default=1):  Set to 0 to suppress Amp-related output.
        min_loss_scale (float, default=None):  Sets a floor for the loss scale values that can be chosen by dynamic
            loss scaling.  The default value of None means that no floor is imposed.
            If dynamic loss scaling is not used, `min_loss_scale` is ignored.
        max_loss_scale (float, default=2.**24):  Sets a ceiling for the loss scale values that can be chosen by
            dynamic loss scaling.  If dynamic loss scaling is not used, `max_loss_scale` is ignored.

    More details about amp_config refer to `amp_config <https://nvidia.github.io/apex/amp.html?highlight=apex%20amp>`_.

    Returns:
        A tuple (model, optimizer).
    """
    import apex.amp as apex_amp
    model, optimizer = apex_amp.initialize(model, optimizer, **amp_config)
    optimizer = ApexAMPOptimizer(optimizer)
    return model, optimizer


__all__ = ['convert_to_apex_amp', 'ApexAMPOptimizer']
