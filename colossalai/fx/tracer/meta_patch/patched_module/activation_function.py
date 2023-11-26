import torch

from ...registry import meta_patched_module


@meta_patched_module.register(torch.nn.ReLU)
@meta_patched_module.register(torch.nn.Sigmoid)
@meta_patched_module.register(torch.nn.GELU)
@meta_patched_module.register(torch.nn.Tanh)
@meta_patched_module.register(torch.nn.ReLU6)
@meta_patched_module.register(torch.nn.PReLU)
def torch_nn_non_linear_act(self, input):
    return torch.empty(input.shape, device="meta")
