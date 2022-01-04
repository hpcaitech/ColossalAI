import torch.nn as nn
from colossalai.registry import LOSSES
import torch


@LOSSES.register_module
class MixupLoss(nn.Module):
    def __init__(self, loss_fn_cls):
        super().__init__()
        self.loss_fn = loss_fn_cls()

    def forward(self, inputs, targets_a, targets_b, lam):
        return lam * self.loss_fn(inputs, targets_a) + (1 - lam) * self.loss_fn(inputs, targets_b)


class MixupAccuracy(nn.Module):
    def forward(self, logits, targets):
        targets = targets['targets_a']
        preds = torch.argmax(logits, dim=-1)
        correct = torch.sum(targets == preds)
        return correct
