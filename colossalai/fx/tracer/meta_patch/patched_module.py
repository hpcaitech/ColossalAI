import torch
from .registry import meta_patched_module


@meta_patched_module.register(torch.nn.Linear)
def torch_nn_linear(self, input):
    return torch.empty(input.shape[:-1] + (self.out_features,), device="meta")
