import torch
import torch.nn as nn
from coati.models.utils import compute_reward, normalize
from torch.nn import Module

from applications.Chat.coati.models.base import Actor
from colossalai.utils import get_current_device

from .base import Experience, ExperienceMaker


class NaiveExperienceMaker(ExperienceMaker):
    """
    Naive experience maker.
    """

    def __init__(self,
                 actor: Actor,
                 critic: Module,
                 reward_model: Module,
                 initial_model: Actor,
                 kl_coef: float = 0.1,
                 offload: bool = False,
                 is_colossalai_strategy: bool = False) -> None:
        super().__init__(actor, critic, reward_model, initial_model, kl_coef)
        self.offload = offload
        self.is_colossalai_strategy = is_colossalai_strategy

    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        if not self.is_colossalai_strategy:
            self.actor.to(get_current_device())
        sequences, attention_mask, action_mask = self.actor.generate(input_ids,
                                                                     return_action_mask=True,
                                                                     **generate_kwargs)
        num_actions = action_mask.size(1)

        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        if self.offload:
            self.actor.to('cpu')
        if not self.is_colossalai_strategy:
            self.initial_model.to(get_current_device())
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
        if self.offload:
            self.initial_model.to('cpu')
        if not self.is_colossalai_strategy:
            self.critic.to(get_current_device())
        value = self.critic(sequences, action_mask, attention_mask)
        if self.offload:
            self.critic.to('cpu')
        if not self.is_colossalai_strategy:
            self.reward_model.to(get_current_device())
        r = self.reward_model(sequences, attention_mask)
        if self.offload:
            self.reward_model.to('cpu')
        reward = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask)

        advantage = reward - value
        # TODO(ver217): maybe normalize adv
        if advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)

        return Experience(sequences, action_log_probs, value, reward, advantage, attention_mask, action_mask)
