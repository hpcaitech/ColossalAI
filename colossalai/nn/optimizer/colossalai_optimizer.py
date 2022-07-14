import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from colossalai.utils import clip_grad_norm_fp32


class ColossalaiOptimizer(Optimizer):

    def __init__(self, optim: Optimizer):
        self.optim = optim

    @property
    def param_groups(self):
        return self.optim.param_groups

    @property
    def defaults(self):
        return self.optim.defaults

    def add_param_group(self, *args, **kwargs):
        return self.optim.add_param_group(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.optim.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        self.optim.zero_grad(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.optim.load_state_dict(*args, **kwargs)

    def state_dict(self):
        return self.optim.state_dict()

    def backward(self, loss: Tensor):
        loss.backward()

    def backward_by_grad(self, tensor: Tensor, grad: Tensor):
        torch.autograd.backward(tensors=tensor, grad_tensors=grad)

    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        if max_norm > 0.0:
            clip_grad_norm_fp32(model.parameters(), max_norm)
