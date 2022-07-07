from curses import meta
import operator
import torch
from .registry import meta_patched_function


@meta_patched_function.register(operator.getitem)
def operator_getitem(a, b):
    # copied from huggingface.utils.fx
    def to_concrete(t):
        if isinstance(t, torch.Tensor):
            concrete = torch.ones_like(t, device="cpu")
            if concrete.dtype in [torch.float16, torch.float32, torch.float64, torch.int32]:
                concrete = concrete.to(torch.int64)
            return concrete
        return t

    if isinstance(a, torch.Tensor):
        # TODO: infer shape without performing the computation.
        if isinstance(b, tuple):
            b = tuple(map(to_concrete, b))
        else:
            b = to_concrete(b)
        return operator.getitem(torch.empty_like(a, device="cpu"), b).to("meta")
    return operator.getitem(a, b)


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


@meta_patched_function.register(torch.abs)
def torch_abs(input, *, out=None):
    assert out is None, 'out is not supported yet'
    return torch.empty(input.shape, device='meta')


@meta_patched_function.register(torch.nn.functional.relu)
def torch_nn_func_relu(input, inplace=False):
    return torch.empty(input.shape, device='meta')


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


@meta_patched_function.register(torch.nn.functional.embedding)
def torch_nn_functional_embedding(input,
                                  weight,
                                  padding_idx=None,
                                  max_norm=None,
                                  norm_type=2.0,
                                  scale_grad_by_freq=False,
                                  sparse=False):
    return torch.empty(*input.shape, weight.shape[-1], device="meta")


@meta_patched_function.register(torch.bmm)
def torch_bmm(input, mat2, *, out=None):
    if out is not None:
        raise ValueError("Don't support in-place abs for MetaTensor analysis")
    batch_size, n, m = input.shape
    _, _, p = mat2.shape
    return torch.empty(batch_size, n, p, device="meta")


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


@meta_patched_function.register(torch.nn.functional.layer_norm)
def torch_nn_func_layernorm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    return torch.empty(input.shape, device='meta')


@meta_patched_function.register(torch.nn.functional.batch_norm)
def torch_nn_func_batchnorm(input,
                            running_mean,
                            running_var,
                            weight=None,
                            bias=None,
                            training=False,
                            momentum=0.1,
                            eps=1e-05):
    return torch.empty(input.shape, device='meta')


@meta_patched_function.register(torch.var_mean)
def torch_var_mean(input, dim, unbiased=True, keepdim=False, *, out=None):
    assert out is None, 'saving to out is not supported yet'
    var = torch.empty(1).squeeze(0).to('meta')
    mean = torch.empty(1).squeeze(0).to('meta')
    return var, mean


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
