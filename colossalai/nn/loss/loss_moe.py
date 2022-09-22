from colossalai.registry import LOSSES
from torch.nn.modules.loss import _Loss


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

    def forward(self, output_dict, *args, **kwargs):
        """
        The ``args`` and ``kwargs`` should at least include parameters below:
        ::
            target (:class:`torch.tensor`): Ground truth class indices or class probabilities.

        Args:
            output_dict: The dictionary should have at least two keys. They are "input" and "moe_loss", meaning
                the input of the loss function and the total loss generated in MoE modules respectively.

        Note:
            The ``args`` and ``kwargs`` may include different parameters varying with different loss function.
        """
        assert "input" in output_dict
        assert "moe_loss" in output_dict
        main_loss = self.loss_fn(output_dict["input"], *args, **kwargs)
        aux_loss = output_dict["moe_loss"]
        return main_loss + self.aux_weight * aux_loss
