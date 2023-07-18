from typing import Union

import torch.nn as nn

from .actor import Actor
from .critic import Critic
from .reward_model import RewardModel


def get_base_model(model: Union[Actor, Critic, RewardModel]) -> nn.Module:
    """Get the base model of our wrapper classes.
    For Actor, Critic and RewardModel, return ``model.model``, 
    it's usually a ``transformers.PreTrainedModel``.

    Args:
        model (nn.Module): model to get base model from

    Returns:
        nn.Module: the base model
    """
    assert isinstance(model, (Actor, Critic, RewardModel)), \
        f'Expect Actor, Critic or RewardModel, got {type(model)}, use unwrap_model first.'
    return model.model


__all__ = ['Actor', 'Critic', 'RewardModel', 'get_base_model']
