import torch
import torch.nn as nn
from torch.optim import Optimizer
from colossalai.amp.naive_amp import NaiveAMPModel
from colossalai.utils import is_no_pp_or_last_stage
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode

from .zero_redundancy_optimizer_level_2 import ZeroRedundancyOptimizer_Level_2
from .zero_redundancy_optimizer_level_3 import ZeroRedundancyOptimizer_Level_3


def convert_to_zero(model: nn.Module,
                    optimizer: Optimizer,
                    level: int,
                    zero_config):
    assert level == 2 or level == 3, 'Only ZERO Optimizer Level 2 and 3 are provided'
    if level == 2:
        if is_no_pp_or_last_stage():
            model = NaiveAMPModel(model, output_to_fp32=True)
        else:
            model = NaiveAMPModel(model, output_to_fp32=False)

    if level == 2:
        optimizer = ZeroRedundancyOptimizer_Level_2(
            init_optimizer=optimizer, **zero_config)
    else:
        optimizer = ZeroRedundancyOptimizer_Level_3(
            init_optimizer=optimizer, module=model, **zero_config)
    return model, optimizer


def zero3_model_context(dtype=torch.half):
    assert dtype == torch.half or dtype == torch.float, f'Invalid dtype, except torch.half or torch.float, got {dtype}'
    import deepspeed
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "offload_param": getattr(gpc.config.zero, 'offload_param_config', None),
            "offload_optimizer": getattr(gpc.config.zero, 'offload_optimizer_config'),
        },
        "aio": getattr(gpc.config.zero, 'aio_config', None)
    }
    remote_device = getattr(
        ds_config['zero_optimization']['offload_param'], 'device', None)
    pin_memory = getattr(
        ds_config['zero_optimization']['offload_param'], 'pin_memory', False)
    return deepspeed.zero.Init(data_parallel_group=gpc.get_group(ParallelMode.DATA),
                               remote_device=remote_device,
                               config_dict_or_path=ds_config,
                               pin_memory=pin_memory,
                               dtype=dtype)


__all__ = ['convert_to_zero', 'ZeroRedundancyOptimizer_Level_2',
           'ZeroRedundancyOptimizer_Level_3', 'zero3_model_context']
