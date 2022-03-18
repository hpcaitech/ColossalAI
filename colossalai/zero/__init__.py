from typing import Tuple

import torch
import torch.nn as nn
from colossalai.amp.naive_amp import NaiveAMPModel
from colossalai.logging import get_dist_logger
from colossalai.zero.sharded_model.sharded_model_v2 import ShardedModelV2
from colossalai.zero.sharded_optim.sharded_optim_v2 import ShardedOptimizerV2
from torch.optim import Optimizer

from .sharded_model import ShardedModel
from .sharded_optim import ShardedOptimizer


def convert_to_zero_v2(model: nn.Module, optimizer: torch.optim.Optimizer, model_config,
                       optimizer_config) -> Tuple[ShardedModelV2, ShardedOptimizerV2]:
    """
    A helper function to integrate the model and optimizer with ZeRO optimizer and off-loading

    :param model: Your model object
    :type model: :class:`torch.nn.Module`
    :param optimizer_config: Your optimizer object
    :type optimizer_config: :class:`dict`

    :return: (model, optimizer)
    :rtype: Tuple
    """

    logger = get_dist_logger('convert_to_zero_v2')

    logger.info(f'optimizer_config is {optimizer_config}')
    if optimizer_config is None:
        optimizer_config = dict()
    logger.info(f'model_config is {model_config}')
    if model_config is None:
        model_config = dict()

    zero_model = ShardedModelV2(model, **model_config)
    zero_optimizer = ShardedOptimizerV2(zero_model, optimizer, **optimizer_config)
    return zero_model, zero_optimizer


def convert_to_zero(model: nn.Module, optimizer: Optimizer, level: int, zero_config: dict):
    """
    A helper function to integrate the model and optimizer with ZeRO optimizer and off-loading

    :param model: Your model object
    :type model: :class:`torch.nn.Module`
    :param optimizer: Your optimizer object
    :type optimizer: :class:`torch.optim.Optimizer`
    :param level: Optimizer level, can be 2 or 3
    :type level: int
    :param zero_config: Configuration for zero
    :type zero_config: dict

    :return: (model, optimizer)
    :rtype: Tuple
    """
    assert 1 <= level <= 3, 'Only ZERO Optimizer Level 1-3 are provided'
    if level in [1, 2]:
        if level == 2:
            if 'partition_grad' in zero_config:
                assert zero_config['partition_grad'], \
                    'Sharded Optimizer requires partition_grad to be True'
            else:
                zero_config['partiton_grad'] = True
        model = NaiveAMPModel(model, output_to_fp32=True)
        optimizer = ShardedOptimizer(optimizer, **zero_config)
    else:
        model = ShardedModel(module=model, **zero_config)
    return model, optimizer


__all__ = ['convert_to_zero', 'ShardedModel', 'ShardedOptimizer']
