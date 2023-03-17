from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from chatgpt.models.base import Actor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B)
    reward: (B)
    advatanges: (B)
    attention_mask: (B, S)
    action_mask: (B, A)

    "A" is the number of actions.
    """
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    reward: torch.Tensor
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
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self


class ExperienceMaker(ABC):

    def __init__(self,
                 actor: Actor,
                 critic: nn.Module,
                 reward_model: nn.Module,
                 initial_model: Actor,
                 kl_coef: float = 0.1) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.kl_coef = kl_coef

    @abstractmethod
    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs) -> Experience:
        pass
