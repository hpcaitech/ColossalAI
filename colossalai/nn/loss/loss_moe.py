import torch.nn as nn
from colossalai.registry import LOSSES
from torch.nn.modules.loss import _Loss
from colossalai.context.moe_context import MOE_CONTEXT


@LOSSES.register_module
class MoeCrossEntropyLoss(_Loss):
    r"""torch.nn.CrossEntropyLoss added with auxiliary loss.

    Args:
        input (:class:`torch.tensor`): Predicted unnormalized scores (often referred to as logits).
        target (:class:`torch.tensor`): Ground truth class indices or class probabilities.
        aux_weight (float, optional): Weight of auxiliary loss in total loss.Defaults 0.01.

    The ``args`` and ``kwargs`` should include parameters below:
    ::

        weight (Tensor, optional)
        size_average (bool, optional)
        ignore_index (int, optional)
        reduce (bool, optional)
        reduction (str, optional)
        label_smoothing (float, optional)

    More details about ``args``, ``kwargs`` and ``torch.nn.functional.cross_entropy`` could be found in
    `Cross_entropy <https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy>`_.
    """

    def __init__(self, aux_weight: float = 0.01, *args, **kwargs):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(*args, **kwargs)
        self.aux_weight = aux_weight

    def forward(self, *args):
        """
        The ``args`` should at least include parameters below:
        ::

            input (:class:`torch.tensor`): Predicted unnormalized scores (often referred to as logits).
            target (:class:`torch.tensor`): Ground truth class indices or class probabilities.

        More details about ``args``, ``kwargs`` and ``torch.nn.functional.cross_entropy`` could be found in
        `Cross_entropy <https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy>`_.
        """
        main_loss = self.loss(*args)
        aux_loss = MOE_CONTEXT.get_loss()
        return main_loss + self.aux_weight * aux_loss


@LOSSES.register_module
class MoeLoss(_Loss):
    """A wrapper class for any loss module to add with auxiliary loss.

    Args:
        aux_weight (float): Weight of auxiliary loss in total loss.
        loss_fn (``Callable``): Loss function.
        args (list): Args in loss function.
        kwargs (dict): Kwargs in loss function
    """

    def __init__(self, aux_weight: float, loss_fn, *args, **kwargs):
        super().__init__()
        self.loss_fn = loss_fn(*args, **kwargs)
        self.aux_weight = aux_weight

    def forward(self, *args, **kwargs):
        """
        The ``args`` and ``kwargs`` should at least include parameters below:
        ::

            input (:class:`torch.tensor`): Predicted unnormalized scores (often referred to as logits).
            target (:class:`torch.tensor`): Ground truth class indices or class probabilities.

        Note:
            The ``args`` and ``kwargs`` may include different parameters varying with different loss function.
        """
        main_loss = self.loss_fn(*args, **kwargs)
        aux_loss = MOE_CONTEXT.get_loss()
        return main_loss + self.aux_weight * aux_loss
