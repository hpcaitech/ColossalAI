import torch
from ..registry import meta_patched_function


@meta_patched_function.register(torch.matmul)
def torch_matmul(input, other, *, out=None):
    # copied from huggingface.utils.fx
    d1 = input.dim()
    d2 = other.dim()
    shape = None
    if d1 == 1 and d2 == 1:
        shape = None
    elif d1 == 2 and d2 == 2:
        shape = (input.size(0), other.size(1))
    elif d1 == 1 and d2 == 2:
        shape = (other.size(1),)
    elif d1 == 2 and d1 == 1:
        shape = (input.size(0),)
    else:
        max_length = max(input.dim(), other.dim())
        shape1 = list(input.shape)
        shape2 = list(other.shape)
        if d1 == 1:
            shape1 = [1] + shape1
        if d2 == 1:
            shape2.append(1)
        shape1 = [-1] * (max_length - d1) + list(input.shape)
        shape2 = [-1] * (max_length - d2) + list(other.shape)
        shape = []
        for i in range(max_length):
            shape.append(max(shape1[i], shape2[i]))
        shape[-2] = shape1[-2]
        shape[-1] = shape2[-1]
        if d1 == 1:
            shape.pop(-2)
        if d2 == 1:
            shape.pop(-1)
    if shape is None:
        return torch.tensor(0.0, device="meta")
    return torch.empty(*shape, device="meta")


@meta_patched_function.register(torch.abs)
def torch_abs(input, *, out=None):
    assert out is None, 'out is not supported yet'
    return torch.empty(input.shape, device='meta')


@meta_patched_function.register(torch.bmm)
def torch_bmm(input, mat2, *, out=None):
    if out is not None:
        raise ValueError("Don't support in-place abs for MetaTensor analysis")
    batch_size, n, m = input.shape
    _, _, p = mat2.shape
    return torch.empty(batch_size, n, p, device="meta")


@meta_patched_function.register(torch.var_mean)
def torch_var_mean(input, dim, unbiased=True, keepdim=False, *, out=None):
    assert out is None, 'saving to out is not supported yet'
    var = torch.empty(1).squeeze(0).to('meta')
    mean = torch.empty(1).squeeze(0).to('meta')
    return var, mean
