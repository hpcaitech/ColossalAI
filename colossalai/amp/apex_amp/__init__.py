from .apex_amp import ApexAMPOptimizer
import torch.nn as nn
from torch.optim import Optimizer
import apex.amp as apex_amp


def convert_to_apex_amp(model: nn.Module,
                        optimizer: Optimizer,
                        amp_config):
    model, optimizer = apex_amp.initialize(model, optimizer, **amp_config)
    optimizer = ApexAMPOptimizer(optimizer)
    return model, optimizer


__all__ = ['convert_to_apex_amp', 'ApexAMPOptimizer']
