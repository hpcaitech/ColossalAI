import torch.nn as nn
from torch.optim import Optimizer
from colossalai.amp.naive_amp import NaiveAMPModel
from colossalai.utils import is_no_pp_or_last_stage

from .zero_redundancy_optimizer_level_2 import ZeroRedundancyOptimizer_Level_2
from .zero_redundancy_optimizer_level_3 import ZeroRedundancyOptimizer_Level_3


def convert_to_zero(model: nn.Module,
                    optimizer: Optimizer,
                    level: int,
                    zero_config):
    assert level == 2 or level == 3, 'Only ZERO Optimizer Level 2 and 3 are provided'

    if is_no_pp_or_last_stage():
        model = NaiveAMPModel(model, output_to_fp32=True)
    else:
        model = NaiveAMPModel(model, output_to_fp32=False)

    if level == 2:
        optimizer = ZeroRedundancyOptimizer_Level_2(init_optimizer=optimizer, **zero_config)
    else:
        optimizer = ZeroRedundancyOptimizer_Level_3(init_optimizer=optimizer, module=model, **zero_config)
    return model, optimizer


__all__ = ['convert_to_zero', 'ZeroRedundancyOptimizer_Level_2', 'ZeroRedundancyOptimizer_Level_3']
