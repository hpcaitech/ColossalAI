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

    Args:
        model (:class:`torch.nn.Module`): your model object.
        optimizer (:class:`torch.optim.Optimizer`): your optimizer object
        criterion (:class:`torch.nn.modules.loss._Loss`, optional): your loss function object
        **amp_config (:class:`colossalai.context.Config` or dict, optional): configuration for torch mode amp.
            The ``amp_config`` should contain ``init_scale``, ``growth_factor``, ``backoff_factor``,
            ``growth_interval`` and ``enabled``.

        init_scale (float, optional, default=2.**16):  Initial scale factor.
        growth_factor (float, optional, default=2.0):  Factor by which the scale is multiplied during
            :meth:`update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
        backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
            :meth:`update` if inf/NaN gradients occur in an iteration.
        growth_interval (int, optional, default=2000):  Number of consecutive iterations without inf/NaN gradients
            that must occur for the scale to be multiplied by ``growth_factor``.
        enabled (bool, optional, default=True):  If ``False``, disables gradient scaling. :meth:`step` simply
            invokes the underlying ``optimizer.step()``, and other methods become no-ops.

    Returns:
        A tuple (model, optimizer, criterion)
    """
    model = TorchAMPModel(model)
    if amp_config is None:
        amp_config = dict()
    optimizer = TorchAMPOptimizer(optimizer, **amp_config)
    if criterion:
        criterion = TorchAMPLoss(criterion)
    return model, optimizer, criterion


__all__ = ['convert_to_torch_amp', 'TorchAMPModel', 'TorchAMPLoss', 'TorchAMPOptimizer']
