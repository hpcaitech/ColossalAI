from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from coati.models.base import Actor, Critic, RewardModel


@dataclass
class Experience:
    """Experience is a batch of data.
    Left padding for sequences is applied.

    "B" is the batch size.
    "S" is the sequence length.
    "A" is the number of actions.
    "C" is the chunk size.
    "N" is the number of MDP steps.
    NOTE: N = A / C, each Experience contains N MDP steps ([s0, a0], [s1, a1], ...),
        sequences = |pad|prompt|a0|a1|a2|...|pad|,
        s0 = prompt, s1 = prompt + a0, s2 = prompt + a0 + a1, ...
    FIXME(cwher): store N steps in a Experience can be computationally efficient,
        but may be different from uniform sampling (shuffle all steps and sample).

    Shapes of each tensor:
        sequences: (B, S)
        attention_mask: (B, S)
        action_mask: (B, A)
        step_mask: (B, N)
        action_log_probs: (B, A)
        values: (B, N), output of old critic model
        returns: (B, N), result of GAE
        advantages: (B, N), result of GAE

    e.g.,
        sequences = |pad|prompt|response|pad|
        attention_mask = |0|1|1|0|
        action_mask = |1|0| (for response)

    NOTE: `Experience` are split into `BufferItem`s when added to buffer.
    """

    sequences: torch.Tensor
    attention_mask: torch.LongTensor
    action_mask: torch.BoolTensor
    step_mask: torch.BoolTensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.action_mask = self.action_mask.to(device)
        self.step_mask = self.step_mask.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.advantages = self.advantages.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        self.action_mask = self.action_mask.pin_memory()
        self.step_mask = self.step_mask.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        self.values = self.values.pin_memory()
        self.returns = self.returns.pin_memory()
        self.advantages = self.advantages.pin_memory()
        return self


class ExperienceMaker(ABC):
    def __init__(self, actor: Actor, critic: Critic, reward_model: RewardModel, initial_model: Actor) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.initial_model = initial_model

    @abstractmethod
    def make_experience(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **generate_kwargs) -> Experience:
        pass
