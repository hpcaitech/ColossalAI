import torch.nn as nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from colossalai.context import Config
from .torch_amp import TorchAMPOptimizer, TorchAMPModel, TorchAMPLoss


def convert_to_torch_amp(model: nn.Module,
                         optimizer: Optimizer,
                         criterion: _Loss,
                         amp_config: Config):
    model = TorchAMPModel(model)
    optimizer = TorchAMPOptimizer(optimizer, **amp_config)
    criterion = TorchAMPLoss(criterion)
    return model, optimizer, criterion


__all__ = ['convert_to_torch_amp', 'TorchAMPModel', 'TorchAMPLoss', 'TorchAMPOptimizer']
