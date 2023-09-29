import torch
from torch import nn

from colossalai.legacy.nn.layer.parallel_2p5d import reduce_by_batch_2p5d, split_batch_2p5d

from ._utils import calc_acc


class Accuracy2p5D(nn.Module):
    """Accuracy for 2p5D parallelism"""

    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        """Calculate the accuracy of predicted labels.

        Args:
            logits (:class:`torch.tensor`): Predicted labels.
            targets (:class:`torch.tensor`): True labels from data.

        Returns:
            float: the accuracy of prediction.
        """
        with torch.no_grad():
            targets = split_batch_2p5d(targets)
            correct = calc_acc(logits, targets)
            correct = reduce_by_batch_2p5d(correct)
        return correct
