import torch

from ...registry import meta_patched_module


@meta_patched_module.register(torch.nn.Embedding)
def torch_nn_embedding(self, input):
    result_shape = input.shape + (self.embedding_dim,)
    return torch.empty(result_shape, device='meta')
