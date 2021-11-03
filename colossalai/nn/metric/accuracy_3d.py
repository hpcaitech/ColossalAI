from colossalai.constants import INPUT_GROUP_3D, WEIGHT_GROUP_3D
from colossalai.nn.layer.parallel_3d import reduce_by_batch_3d, split_batch_3d
from colossalai.nn.layer.parallel_3d._utils import (get_depth_from_env, get_last_group, get_parallel_mode_from_env)
from torch import nn

from ._utils import calc_acc


class Accuracy3D(nn.Module):
    def __init__(self):
        #  input_parallel_mode, weight_parallel_mode):
        super().__init__()
        self.depth = get_depth_from_env()
        self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        self.output_parallel_mode = get_last_group(self.input_parallel_mode, self.weight_parallel_mode)

    def forward(self, logits, targets):
        targets = split_batch_3d(targets, self.input_parallel_mode, self.weight_parallel_mode)

        # batch_size = targets.size(0)

        # j = gpc.get_local_rank(self.input_parallel_mode)
        # i = gpc.get_local_rank(self.weight_parallel_mode)
        # target = torch.chunk(target, self.depth, dim=0)[i]
        # target = torch.chunk(target, self.depth, dim=0)[j]

        # logits = all_gather(logits, -1, self.output_parallel_mode)
        # logits = torch.cat(logits, dim=-1)
        # prediction = torch.argmax(logits, dim=-1)
        # correct = torch.sum(prediction == targets)
        correct = calc_acc(logits, targets)

        # dist.all_reduce(correct, group=gpc.get_group(self.input_parallel_mode))
        # dist.all_reduce(correct,
        #                 group=gpc.get_group(self.weight_parallel_mode))
        correct = reduce_by_batch_3d.apply(correct, self.input_parallel_mode, self.weight_parallel_mode)

        return correct
