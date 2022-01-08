import torch.nn as nn
from colossalai.registry import LOSSES
from torch.nn.modules.loss import _Loss
from colossalai.global_variables import moe_env


@LOSSES.register_module
class MoeCrossEntropyLoss(_Loss):
    """torch.nn.CrossEntropyLoss added with auxiliary loss.
    """
    def __init__(self, aux_weight: float = 0.01, *args, **kwargs):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(*args, **kwargs)
        self.aux_weight = aux_weight

    def forward(self, *args):
        main_loss = self.loss(*args)
        aux_loss = moe_env.get_loss()
        return main_loss + self.aux_weight * aux_loss


@LOSSES.register_module
class MoeLoss(_Loss):
    """A wrapper class for any loss module to add with auxiliary loss.
    """
    def __init__(self, aux_weight: float, loss_fn, *args, **kwargs):
        super().__init__()
        self.loss_fn = loss_fn(*args, **kwargs)
        self.aux_weight = aux_weight

    def forward(self, *args, **kwargs):
        main_loss = self.loss_fn(*args, **kwargs)
        aux_loss = moe_env.get_loss()
        return main_loss + self.aux_weight * aux_loss
