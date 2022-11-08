import torch

from ...registry import meta_patched_module


@meta_patched_module.register(torch.nn.Linear)
def torch_nn_linear(self, input):
    last_dim = input.shape[-1]
    assert last_dim == self.in_features, f'Expected hidden size {self.in_features} but got {last_dim} for the torch.nn.Linear patch'
    return torch.empty(input.shape[:-1] + (self.out_features,), device="meta")
