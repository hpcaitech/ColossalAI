import torch
from ..registry import meta_patched_function


@meta_patched_function.register(torch.arange)
def torch_arange(*args, **kwargs):
    n = len(args)
    step = 1
    if n == 1:
        start = 0
        end = args[0]
    elif n == 2:
        start, end = args
    else:
        start, end, step = args
    if isinstance(start, float):
        start = int(start)
    if isinstance(end, float):
        start = int(end)
    if isinstance(step, float):
        step = int(step)
    step = kwargs.get("step", step)
    dtype = kwargs.get("dtype")
    return torch.empty((end - start) // step, dtype=dtype, device="meta")


@meta_patched_function.register(torch.where)
def torch_where(condition, x, y):
    # torch.where returns the broadcasted tensor of condition, x, and y,
    # so hack it by using addition
    return condition.to(device="meta") + x.to(device="meta") + y.to(device="meta")


@meta_patched_function.register(torch.Tensor.repeat)
def torch_tensor_repeat(self, *sizes):
    shape = list(self.shape)
    for i, x in enumerate(sizes):
        shape[i] *= x
    return torch.empty(shape, device="meta")


@meta_patched_function.register(torch.index_select)
def torch_index_select(input, dim, index, *, out=None):
    shape = list(input.shape)
    shape[dim] = len(index)
    return torch.empty(*shape, device="meta")


@meta_patched_function.register(torch.Tensor.index_select)
def torch_tensor_index_select(self, dim, index):
    return torch_index_select(self, dim, index)


@meta_patched_function.register(torch.squeeze)
def torch_squeeze(input, dim=None):
    shape = list(input.shape)
    if dim is not None:
        if dim < 0:
            dim = input.dim() + dim
        if shape[dim] == 1:
            shape.pop(dim)
    else:
        new_shape = []
        for dim_value in shape:
            if dim_value == 1:
                continue
            new_shape.append(dim_value)
        shape = new_shape
    return torch.empty(shape, device="meta")


@meta_patched_function.register(torch.Tensor.squeeze)
def torch_tensor_squeeze(self, dim=None):
    return torch_squeeze(self, dim)


@meta_patched_function.register(torch.unsqueeze)
def torch_unsqueeze(input, dim):
    shape = list(input.shape)
    if dim < 0:
        dim = input.dim() + 1 + dim
    shape.insert(dim, 1)
    return torch.empty(shape, device="meta")


@meta_patched_function.register(torch.Tensor.unsqueeze)
def torch_tensor_unsqueeze(self, dim):
    return torch_unsqueeze(self, dim)


@meta_patched_function.register(torch.cat)
def torch_cat(tensors, dim=None, axis=None, *, out=None):
    if dim is None and axis is None:
        dim = 0
    if dim is None and axis is not None:
        dim = axis
    if dim < 0:
        dim = tensors[0].dim() + dim
    shapes = [t.shape for t in tensors]
    shape = list(shapes[0])
    concatenated_dim = sum(shape[dim] for shape in shapes)
    final_shape = shape[:dim] + [concatenated_dim] + shape[dim + 1:]
    return torch.empty(final_shape, device="meta")


@meta_patched_function.register(torch.roll)
def torch_roll(input, shifts, dims=None):
    return torch.empty(input.shape, device='meta')
