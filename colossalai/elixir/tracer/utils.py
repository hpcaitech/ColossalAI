from collections import OrderedDict
from copy import copy
from typing import Optional, Set

import torch
import torch.nn as nn


def _get_dfs_module_list(module: nn.Module, memo: Optional[Set[nn.Module]] = None, prefix: str = ''):
    """Get a dfs module list of the given module. Its order is same as the order of creations of modules.
    """
    if memo is None:
        memo = set()
    if module not in memo:
        for name, submodule in module._modules.items():
            if submodule is None:
                continue
            submodule_prefix = prefix + ('.' if prefix else '') + name
            for m in _get_dfs_module_list(submodule, memo, submodule_prefix):
                yield m

        memo.add(module)
        yield prefix, module


def _get_shallow_copy_model(model: nn.Module):
    """Get a shallow copy of the given model. Each submodule is different from the original submodule.
    But the new submodule and the old submodule share all attributes.
    """
    old_to_new = dict()
    for name, module in _get_dfs_module_list(model):
        new_module = copy(module)
        new_module._modules = OrderedDict()
        for subname, submodule in module._modules.items():
            if submodule is None:
                continue
            setattr(new_module, subname, old_to_new[submodule])
        old_to_new[module] = new_module
    return old_to_new[model]


def meta_copy(model: nn.Module, meta_fn: callable):
    new_model = _get_shallow_copy_model(model)
    old_parameters = dict()
    old_buffers = dict()

    for (_, old_module), (_, new_module) in \
        zip(_get_dfs_module_list(model), _get_dfs_module_list(new_model)):

        new_module._parameters = OrderedDict()
        for name, param in old_module._parameters.items():
            new_param = None
            if param is not None:
                param_id = id(param)
                if param_id in old_parameters:
                    new_param = old_parameters.get(param_id)
                else:
                    new_param = meta_fn(param)
                    old_parameters[param_id] = new_param
            setattr(new_module, name, new_param)

        new_module._buffers = OrderedDict()
        for name, buffer in old_module._buffers.items():
            new_buffer = None
            if buffer is not None:
                buffer_id = id(buffer)
                if buffer_id in old_buffers:
                    new_buffer = old_buffers.get(buffer_id)
                else:
                    new_buffer = meta_fn(buffer)
                    old_buffers[buffer_id] = new_buffer
            new_module.register_buffer(name, new_buffer)

    return new_model


def get_cuda_allocated():
    return torch.cuda.memory_allocated()


def get_cuda_max_allocated():
    return torch.cuda.max_memory_allocated()


def model_memory_figure(model: nn.Module):
    param_occ = 0
    max_numel = 0
    for name, param in model.named_parameters():
        param_occ += param.numel() * param.element_size()
        max_numel = max(max_numel, param.numel())

    buffer_occ = 0
    for name, buffer in model.named_buffers():
        buffer_occ += buffer.numel() * buffer.element_size()

    return dict(param_occ=param_occ, param_max_numel=max_numel, buffer_occ=buffer_occ)
