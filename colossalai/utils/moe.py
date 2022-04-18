import torch.nn as nn
import torch.distributed as dist
from colossalai.core import global_context as gpc
from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.context import ParallelMode
from .common import is_using_ddp
from typing import Dict, List


def get_moe_epsize_param_dict(model: nn.Module) -> Dict[int, List[nn.Parameter]]:
    """Returns a parameter dictionary, the key of which is the expert parallel
    size of every parameter. Since the parameters in data parallelism is replicated
    in each GPU, we set their ep_size to 1.

    Args:
        model (:class:`torch.nn.Module`): A pyTorch `nn.Module` from which we get dict.
    """
    epsize_param_dict = dict()
    for param in model.parameters():
        if not hasattr(param, 'moe_info'):
            ep_size = 1    # set ep_size to 1 for dp parameters
        else:
            ep_size = param.moe_info.ep_size
        if ep_size not in epsize_param_dict:
            epsize_param_dict[ep_size] = []
        epsize_param_dict[ep_size].append(param)

    return epsize_param_dict


def sync_moe_model_param(model: nn.Module):
    """Make sure model parameters are consistent in MoE parallel context.

    Args:
        model (:class:`torch.nn.Module`): A pyTorch model on whose parameters you check the consistency.
    """
    if is_using_ddp():

        param_dict = get_moe_epsize_param_dict(model)

        # synchrosize the parameters whose dp_group is the whole world
        if 1 in param_dict:
            src_rank = gpc.get_ranks_in_group(ParallelMode.DATA)[0]
            for param in param_dict[1]:
                dist.broadcast(param, src=src_rank, group=gpc.get_group(ParallelMode.DATA))

        for ep_size in param_dict:
            # When ep_size = world_size, communication is not needed
            if ep_size != 1 and ep_size != MOE_CONTEXT.world_size:
                src_rank = dist.get_rank(MOE_CONTEXT.parallel_info_dict[ep_size].ep_group)
                for param in param_dict[ep_size]:
                    dist.broadcast(param, src=src_rank, group=param.moe_info.dp_group)
