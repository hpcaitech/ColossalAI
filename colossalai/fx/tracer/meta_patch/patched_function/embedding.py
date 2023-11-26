import torch

from ...registry import meta_patched_function


@meta_patched_function.register(torch.nn.functional.embedding)
def torch_nn_functional_embedding(
    input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False
):
    return torch.empty(*input.shape, weight.shape[-1], device="meta")
