import torch
from coati.models.utils import compute_reward, normalize

from colossalai.tensor import ColoTensor

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

        sequences, attention_mask, action_mask = self.actor.generate(input_ids,
                                                                     return_action_mask=True,
                                                                     **generate_kwargs)
        num_actions = action_mask.size(1)

        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        value = self.critic(sequences, action_mask, attention_mask)
        # converting to torch.tensor when using gemini strategy
        if isinstance(sequences, ColoTensor):
            torch_sequences = sequences.data.to(torch.cuda.current_device())
            torch_attention_mask = attention_mask.data.to(torch.cuda.current_device())
            base_action_log_probs = self.initial_model(torch_sequences, num_actions, torch_attention_mask)
            r = self.reward_model(torch_sequences, torch_attention_mask)
        else:
            base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
            r = self.reward_model(sequences, attention_mask)
        reward = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask)

        advantage = reward - value
        # TODO(ver217): maybe normalize adv
        if advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)

        return Experience(sequences, action_log_probs, value, reward, advantage, attention_mask, action_mask)
