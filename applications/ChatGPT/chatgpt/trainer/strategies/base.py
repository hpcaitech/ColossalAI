from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Tuple

import torch
import torch.nn as nn
from chatgpt.nn import Actor, Critic, RewardModel
from chatgpt.replay_buffer import ReplayBuffer
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Strategy(ABC):
    """
        Base class for training strategies.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setup_distributed()

    @abstractmethod
    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: Optimizer, **kwargs) -> None:
        pass

    @abstractmethod
    def optimizer_step(self, optimizer: Optimizer, **kwargs) -> None:
        pass

    @abstractmethod
    def setup_distributed(self) -> None:
        pass

    @abstractmethod
    def setup_model(self, model: nn.Module) -> nn.Module:
        pass

    @abstractmethod
    def setup_optimizer(self, optimizer: Optimizer, model: nn.Module) -> Optimizer:
        pass

    @abstractmethod
    def setup_dataloader(self, replay_buffer: ReplayBuffer, pin_memory: bool = False) -> DataLoader:
        pass

    def model_init_context(self):
        return nullcontext()

    def prepare(self, actor: Actor, critic: Critic, reward_model: RewardModel, initial_model: Actor,
                actor_optim: Optimizer,
                critic_optim: Optimizer) -> Tuple[Actor, nn.Module, nn.Module, nn.Module, Optimizer, Optimizer]:
        """Prepare (actor, critic, reward_model, initial_model, actor_optim, critic_optim) based on each strategy.

        Args:
            actor (Actor): actor
            critic (Critic): critc
            reward_model (RewardModel): reward model
            initial_model (Actor): initial model
            actor_optim (Optimizer): actor's optimizer
            critic_optim (Optimizer): critic's optimizer

        Returns:
            Tuple[Actor, nn.Module, nn.Module, nn.Module, Optimizer, Optimizer]: (actor, critic, reward_model, initial_model, actor_optim, critic_optim)
        """
        actor = Actor(self.setup_model(actor.model))
        critic = self.setup_model(critic)
        reward_model = self.setup_model(reward_model)
        initial_model = Actor(self.setup_model(initial_model.model))

        actor_optim = self.setup_optimizer(actor_optim, actor.model)
        critic_optim = self.setup_optimizer(critic_optim, critic)

        return actor, critic, reward_model, initial_model, actor_optim, critic_optim

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """Useful for saving state dict. As actor is wrapped by Actor class again in `prepare()`, we should unwrap it before saving.

        Args:
            model (nn.Module): an actor or a critic
        """
        if isinstance(model, Actor):
            return model.model
        return model

    @staticmethod
    def _unwrap_actor(actor: Actor) -> nn.Module:
        """Get `actor.model` from a wrapped (by `prepare()`) actor. Useful for getting original huggingface model.

        Args:
            actor (Actor): a wrapped actor
        """
        return Strategy._unwrap_model(actor)

    @abstractmethod
    def save_model(self, model: nn.Module, path: str, only_rank0: bool = True) -> None:
        pass

    @abstractmethod
    def load_model(self, model: nn.Module, path: str, map_location: Any = None) -> None:
        pass

    @abstractmethod
    def save_optimizer(self, optimizer: Optimizer, path: str, only_rank0: bool = True) -> None:
        pass

    @abstractmethod
    def load_optimizer(self, optimizer: Optimizer, path: str, map_location: Any = None) -> None:
        pass
