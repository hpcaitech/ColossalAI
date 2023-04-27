import torch.nn as nn

from .actor import Actor
from .critic import Critic
from .reward_model import RewardModel


def get_base_model(model: nn.Module) -> nn.Module:
    """Get the base model of our wrapper classes.
    For Actor, it's base model is ``actor.model`` and it's usually a ``transformers.PreTrainedModel``.
    For Critic and RewardModel, it's base model is itself.

    Args:
        model (nn.Module): model to get base model from

    Returns:
        nn.Module: the base model
    """
    if isinstance(model, Actor):
        return model.get_base_model()
    return model


__all__ = ['Actor', 'Critic', 'RewardModel', 'get_base_model']
