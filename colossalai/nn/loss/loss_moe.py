import torch.nn as nn
from colossalai.registry import LOSSES
from torch.nn.modules.loss import _Loss
from colossalai.global_variables import moe_env


@LOSSES.register_module
class MoeCrossEntropyLoss(_Loss):
    """torch.nn.CrossEntropyLoss added with auxiliary loss.

    :param reduction: whether to average the loss, defaults to True
    :type reduction: bool, optional
    :param args: Args for torch.nn.functional.cross_entropy
    :param kwargs: Kwargs for torch.nn.functional.cross_entropy

    the parameters args and kwargs could contain: [weight (Tensor, optional), size_average (bool, optional),
    ignore_index (int, optional), label_smoothing (float, optional)]

    more details about args, kwargs and torch.nn.functional.cross_entropy could be found in
    https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy
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

    :param aux_weight: Weight of auxiliary loss in total loss
    :param loss_fn: Loss function
    :param args: Args in the loss function loss_fn
    :param kwargs: Kwargs in the loss function loss_fn

    :type aux_weight: float
    :type loss_fn: Callable
    """
    def __init__(self, aux_weight: float, loss_fn, *args, **kwargs):
        super().__init__()
        self.loss_fn = loss_fn(*args, **kwargs)
        self.aux_weight = aux_weight

    def forward(self, *args, **kwargs):
        main_loss = self.loss_fn(*args, **kwargs)
        aux_loss = moe_env.get_loss()
        return main_loss + self.aux_weight * aux_loss
