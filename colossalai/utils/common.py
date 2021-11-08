#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.distributed as dist

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc


def print_rank_0(msg: str, logger=None):
    '''Print messages and save logs(optional). This is executed only if you are the rank-0 gpu.

    :param msg: A str message to output
    :param logger: python logger object, defaults to None
    '''
    if gpc.get_global_rank() == 0:
        if logger is None:
            print(msg, flush=True)
        else:
            logger.info(msg)
            # print(msg, flush=True)


def sync_model_param_in_dp(model):
    '''Make sure data parameters are consistent during Data Parallel Mode

    :param model: A pyTorch nn.model on whose parameters you check the consistency
    '''
    
    if gpc.is_initialized(ParallelMode.DATA) and gpc.get_world_size(ParallelMode.DATA) > 1:
        for param in model.parameters():
            ranks = gpc.get_ranks_in_group(ParallelMode.DATA)
            dist.broadcast(param, src=ranks[0], group=gpc.get_group(ParallelMode.DATA))

def is_dp_rank_0():
    return not gpc.is_initialized(ParallelMode.DATA) or gpc.is_first_rank(ParallelMode.DATA)

def is_tp_rank_0():
    return not gpc.is_initialized(ParallelMode.TENSOR) or gpc.is_first_rank(ParallelMode.TENSOR)

def is_no_pp_or_last_stage():
    return not gpc.is_initialized(ParallelMode.PIPELINE) or gpc.is_last_rank(ParallelMode.PIPELINE)