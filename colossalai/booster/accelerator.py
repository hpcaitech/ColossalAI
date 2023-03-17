import torch
import torch.nn as nn

__all__ = ['Accelerator']


class Accelerator:

    def __init__(self, device: torch.device):
        self.device = device

    def setup_model(self, model: nn.Module) -> nn.Module:
        # TODO: implement this method
        pass
