from typing import Tuple

import torch
import torch.nn as nn

from colossalai.logging import get_dist_logger
from colossalai.zero.sharded_model.sharded_model_v2 import ShardedModelV2
from colossalai.zero.sharded_optim import LowLevelZeroOptimizer, ShardedOptimizerV2

from ..nn.optimizer.zero_optimizer import ZeroOptimizer


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

    logger.info(f'optimizer_config is {optimizer_config}', ranks=[0])
    if optimizer_config is None:
        optimizer_config = dict()
    logger.info(f'model_config is {model_config}', ranks=[0])
    if model_config is None:
        model_config = dict()

    zero_model = ShardedModelV2(model, **model_config)
    zero_optimizer = ShardedOptimizerV2(zero_model, optimizer, **optimizer_config)
    return zero_model, zero_optimizer


__all__ = ['convert_to_zero_v2', 'LowLevelZeroOptimizer', 'ShardedModelV2', 'ShardedOptimizerV2', 'ZeroOptimizer']
