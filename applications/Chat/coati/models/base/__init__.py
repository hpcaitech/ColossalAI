import torch.nn as nn

from .actor import Actor
from .critic import Critic
from .reward_model import RewardModel


def get_base_model(model: nn.Module) -> nn.Module:
    if isinstance(model, Actor):
        return model.get_base_model()
    return model


__all__ = ['Actor', 'Critic', 'RewardModel', 'get_base_model']
