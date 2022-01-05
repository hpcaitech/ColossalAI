import torch
from colossalai.constants import INPUT_GROUP_3D, WEIGHT_GROUP_3D
from colossalai.nn.layer.parallel_3d import reduce_by_batch_3d
from colossalai.nn.layer.parallel_3d._utils import get_parallel_mode_from_env
from torch import nn

from ._utils import calc_acc


class Accuracy3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)

    def forward(self, logits, targets):
        with torch.no_grad():
            correct = calc_acc(logits, targets)
            correct = reduce_by_batch_3d.apply(correct, self.input_parallel_mode, self.weight_parallel_mode)
        return correct
