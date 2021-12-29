import torch
from colossalai.nn.layer.parallel_2d import reduce_by_batch_2d, split_batch_2d
from torch import nn

from ._utils import calc_acc


class Accuracy2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        with torch.no_grad():
            targets = split_batch_2d(targets)
            correct = calc_acc(logits, targets)
            correct = reduce_by_batch_2d.apply(correct)
        return correct
