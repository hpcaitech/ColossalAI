from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from coati.models import Critic, RewardModel
from transformers import PreTrainedModel


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B)
    reward: (B)
    advantages: (B)
    attention_mask: (B, S)
    action_mask: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    reward: torch.Tensor
    kl: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.values = self.values.to(device)
        self.reward = self.reward.to(device)
        self.advantages = self.advantages.to(device)
        self.kl = self.kl.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        self.values = self.values.pin_memory()
        self.reward = self.reward.pin_memory()
        self.advantages = self.advantages.pin_memory()
        self.kl = self.kl.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self


class ExperienceMaker(ABC):
    """
    Base class for experience makers.
    """

    def __init__(
        self, actor: PreTrainedModel, critic: Critic, reward_model: RewardModel, initial_model: PreTrainedModel
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.initial_model = initial_model

    @abstractmethod
    def make_experience(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **generate_kwargs) -> Experience:
        """
        Abstract method to generate an experience.

        Args:
            input_ids (torch.Tensor): The input tensor.
            attention_mask (torch.Tensor): The attention mask tensor.
            **generate_kwargs: Additional keyword arguments for generating the experience.

        Returns:
            Experience: The generated experience.
        """
