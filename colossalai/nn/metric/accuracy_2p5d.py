import torch
from colossalai.nn.layer.parallel_2p5d import reduce_by_batch_2p5d, split_batch_2p5d
from torch import nn

from ._utils import calc_acc


class Accuracy2p5D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        with torch.no_grad():
            targets = split_batch_2p5d(targets)
            correct = calc_acc(logits, targets)
            correct = reduce_by_batch_2p5d.apply(correct)
        return correct
