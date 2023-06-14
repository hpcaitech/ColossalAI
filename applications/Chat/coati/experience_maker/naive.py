import torch
from coati.models.generation import generate_with_actor
from coati.models.utils import calc_action_log_probs, compute_reward, normalize

from .base import Experience, ExperienceMaker


class NaiveExperienceMaker(ExperienceMaker):
    """
    Naive experience maker.
    """

    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        sequences, attention_mask, action_mask = generate_with_actor(self.actor,
                                                                     input_ids,
                                                                     return_action_mask=True,
                                                                     **generate_kwargs)
        num_actions = action_mask.size(1)

        actor_output = self.actor(sequences, attention_mask)
        action_log_probs = calc_action_log_probs(actor_output, sequences, num_actions)
        base_model_output = self.initial_model(sequences, attention_mask)
        base_action_log_probs = calc_action_log_probs(base_model_output, sequences, num_actions)
        value = self.critic(sequences, action_mask, attention_mask)
        r = self.reward_model(sequences, attention_mask)
        reward = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask)

        advantage = reward - value
        # TODO(ver217): maybe normalize adv
        if advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)

        return Experience(sequences, action_log_probs, value, reward, advantage, attention_mask, action_mask)
