from collections import OrderedDict
from copy import copy
from typing import Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.gemini.chunk import Chunk
from colossalai.utils import get_current_device


def get_temp_total_chunk_on_cuda(chunk: Chunk):
    if chunk.is_gathered:
        return chunk.cuda_global_chunk

    if chunk.cuda_shard is not None:
        shard_temp = chunk.cuda_shard
    else:
        shard_temp = chunk.cpu_shard.to(get_current_device())

    total_temp = torch.zeros(chunk.chunk_size, dtype=chunk.dtype, device=get_current_device())
    gather_list = list(torch.chunk(input=total_temp, chunks=chunk.pg_size, dim=0))
    dist.all_gather(tensor_list=gather_list, tensor=shard_temp, group=chunk.torch_pg)

    return total_temp


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


def get_static_torch_model(zero_ddp_model,
                           device=torch.device("cpu"),
                           dtype=torch.float32,
                           only_rank_0=True) -> torch.nn.Module:
    """Get a static torch.nn.Module model from the given ZeroDDP module.
    You should notice that the original ZeroDDP model is not modified.
    Thus, you can use the original model in further training.
    But you should not use the returned torch model to train, this can cause unexpected errors.

    Args:
        zero_ddp_model (ZeroDDP): a zero ddp model
        device (torch.device): the device of the final torch model
        dtype (torch.dtype): the dtype of the final torch model
        only_rank_0 (bool): if True, only rank0 has the coverted torch model

    Returns:
        torch.nn.Module: a static torch model used for saving checkpoints or numeric checks
    """
    from colossalai.nn.parallel import ZeroDDP
    assert isinstance(zero_ddp_model, ZeroDDP)

    state_dict = zero_ddp_model.state_dict(only_rank_0=only_rank_0, strict=False)
    colo_model = zero_ddp_model.module
    torch_model = _get_shallow_copy_model(colo_model)

    if not only_rank_0 or dist.get_rank() == 0:
        # record the mapping relationship between colo parameters and torch parameters
        colo_to_torch = dict()
        for (name, colo_module), (_, torch_module) in \
                zip(_get_dfs_module_list(colo_model), _get_dfs_module_list(torch_model)):
            # clean the parameter list of the new torch module
            torch_module._parameters = OrderedDict()
            for sufix_param_name, param in colo_module.named_parameters(recurse=False):
                # get the full name of the parameter
                full_param_name = name + ('.' if name else '') + sufix_param_name

                if full_param_name not in state_dict:
                    # this means the parameter is shared by multiple modules
                    # we should use colo_to_torch to get the torch parameter created before
                    assert param in colo_to_torch, f"can not find parameter `{full_param_name}` in the GeminiDDP module"
                    torch_param = colo_to_torch[param]
                else:
                    # we meet the parameter the first time, just use the state dict to get the data
                    state_param = state_dict[full_param_name]
                    torch_param = torch.nn.Parameter(state_param.data.to(device=device, dtype=dtype))
                    colo_to_torch[param] = torch_param

                setattr(torch_module, sufix_param_name, torch_param)
    dist.barrier()

    return torch_model
