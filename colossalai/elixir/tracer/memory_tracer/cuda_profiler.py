import gc
from typing import Callable, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from colossalai.elixir.tracer.utils import get_cuda_allocated, meta_copy, model_memory_figure
from colossalai.elixir.utils import print_rank_0

from .memory_tensor import MTensor


def grad_cleaner(grad):
    empty_grad = torch.empty_like(grad.elem)
    grad.elem = None
    empty_grad.storage().resize_(0)
    return empty_grad


def cuda_memory_profiling(model: nn.Module, inp: Dict, step_fn: Callable, dtype=torch.float):
    assert isinstance(inp, dict), 'the example input should be a dictionary'
    print_rank_0(f'You are profiling cuda memory with dtype `{dtype}`')

    def tensor_trans(t: torch.Tensor):
        # set dtype for tensors
        meta_dtype = dtype if t.is_floating_point() else t.dtype
        meta_t = torch.empty_like(t.data, device='meta', dtype=meta_dtype)
        # pack parameters
        if isinstance(t, nn.Parameter):
            meta_t = nn.Parameter(meta_t)
        return meta_t

    # first, transform the model into one dtype
    model = meta_copy(model, tensor_trans)
    # get the memory firgure of the model
    memo_dict = model_memory_figure(model)
    # initialize a empty pool for parameters
    pool = torch.zeros(memo_dict['param_max_numel'], device='cuda', dtype=dtype)

    def tensor_to_cuda(t):
        if isinstance(t, nn.Parameter):
            fake_data = pool[:t.numel()].view(t.shape)
            return nn.Parameter(fake_data)
        else:
            fake_data = torch.zeros(t.shape, device='cuda', dtype=t.dtype)
            return fake_data

    # make all parameters in CUDA and point to a same address
    model = meta_copy(model, tensor_to_cuda)
    # add hooks to clean gradients
    for param in model.parameters():
        param.register_hook(grad_cleaner)

    def input_trans(t):
        if isinstance(t, torch.Tensor):
            cuda_dtype = dtype if t.is_floating_point() else t.dtype
            cuda_t = t.data.clone()
            cuda_t = cuda_t.to(dtype=cuda_dtype, device='cuda')
            cuda_t.requires_grad = t.requires_grad
            return MTensor(cuda_t)
        return t

    inp = tree_map(input_trans, inp)
    # reset all collected peak memory states
    MTensor.reset_peak_memory()
    before_cuda_alc = get_cuda_allocated()

    step_fn(model, inp)

    after_cuda_alc = MTensor.current_peak_memory()
    activation_occ = after_cuda_alc - before_cuda_alc

    return dict(param_occ=memo_dict['param_occ'],
                buffer_occ=memo_dict['buffer_occ'],
                grad_occ=memo_dict['param_occ'],
                activation_occ=activation_occ)
