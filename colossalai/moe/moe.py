from typing import Dict, List

import torch.distributed as dist
import torch.nn as nn

from colossalai.moe.manager import MOE_MANAGER
from colossalai.tensor.moe_tensor.api import get_dp_group, get_dp_group_ranks, get_ep_size, is_moe_tensor


def get_moe_epsize_param_dict(model: nn.Module) -> Dict[int, List[nn.Parameter]]:
    """Returns a parameter dictionary, the key of which is the expert parallel
    size of every parameter. Since the parameters in data parallelism is replicated
    in each GPU, we set their ep_size to 1.

    Args:
        model (:class:`torch.nn.Module`): A pyTorch `nn.Module` from which we get dict.
    """
    epsize_param_dict = dict()
    for param in model.parameters():
        if not is_moe_tensor(param):
            ep_size = 1    # set ep_size to 1 for dp parameters
        else:
            ep_size = get_ep_size(param)
        if ep_size not in epsize_param_dict:
            epsize_param_dict[ep_size] = []
        epsize_param_dict[ep_size].append(param)

    return epsize_param_dict


def sync_moe_model_param(model: nn.Module):
    """Make sure model parameters are consistent in MoE parallel context.

    Args:
        model (:class:`torch.nn.Module`): A pyTorch model on whose parameters you check the consistency.
    """
    param_dict = get_moe_epsize_param_dict(model)

    # synchronize the parameters whose dp_group is the whole world
    if 1 in param_dict:
        for param in param_dict[1]:
            dist.broadcast(param, src=0)

    for ep_size in param_dict:
        # When ep_size = world_size, communication is not needed
        if ep_size != 1 and ep_size != MOE_MANAGER.world_size:
            for param in param_dict[ep_size]:
                src_rank = get_dp_group_ranks(param)[0]
                dist.broadcast(param, src=src_rank, group=get_dp_group(param))
