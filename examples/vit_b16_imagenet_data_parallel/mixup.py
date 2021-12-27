import torch.nn as nn
from colossalai.registry import LOSSES

@LOSSES.register_module
class MixupLoss(nn.Module):
    def __init__(self, loss_fn_cls):
        super().__init__()
        self.loss_fn = loss_fn_cls()

    def forward(self, inputs, *args):
        targets_a, targets_b, lam = args
        return lam * self.loss_fn(inputs, targets_a) + (1 - lam) * self.loss_fn(inputs, targets_b)
