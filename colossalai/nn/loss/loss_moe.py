import torch.nn as nn
from colossalai.registry import LOSSES
from torch.nn.modules.loss import _Loss
from colossalai.context.moe_context import MOE_CONTEXT


@LOSSES.register_module
class MoeCrossEntropyLoss(_Loss):
    """torch.nn.CrossEntropyLoss added with auxiliary loss.

    :param aux_weight: Weight of auxiliary loss in total loss
    :param args: Args in CrossEntropyLoss
    :param kwargs: Kwargs in CrossEntropyLoss

    :type aux_weight: float, optional
    """

    def __init__(self, aux_weight: float = 0.01, *args, **kwargs):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(*args, **kwargs)
        self.aux_weight = aux_weight

    def forward(self, *args):
        main_loss = self.loss(*args)
        aux_loss = MOE_CONTEXT.get_loss()
        return main_loss + self.aux_weight * aux_loss


@LOSSES.register_module
class MoeLoss(_Loss):
    """A wrapper class for any loss module to add with auxiliary loss.

    :param aux_weight: Weight of auxiliary loss in total loss
    :param loss_fn: Loss function
    :param args: Args in loss function
    :param kwargs: Kwargs in loss function

    :type aux_weight: float
    :type loss_fn: Callable
    """

    def __init__(self, aux_weight: float, loss_fn, *args, **kwargs):
        super().__init__()
        self.loss_fn = loss_fn(*args, **kwargs)
        self.aux_weight = aux_weight

    def forward(self, *args, **kwargs):
        main_loss = self.loss_fn(*args, **kwargs)
        aux_loss = MOE_CONTEXT.get_loss()
        return main_loss + self.aux_weight * aux_loss
