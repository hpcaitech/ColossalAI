import torch
import torch.nn.functional as F
from coati.models.generation import generate
from coati.models.utils import calc_action_log_probs, compute_reward

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

        # generate sequences
        sequences = generate(self.actor, input_ids, **generate_kwargs)

        # calculate auxiliary tensors
        attention_mask = None
        pad_token_id = generate_kwargs.get("pad_token_id", None)
        if pad_token_id is not None:
            attention_mask = sequences.not_equal(pad_token_id).to(dtype=torch.long, device=sequences.device)

        input_len = input_ids.size(1)
        eos_token_id = generate_kwargs.get("eos_token_id", None)
        if eos_token_id is None:
            action_mask = torch.ones_like(sequences, dtype=torch.bool)
        else:
            # left padding may be applied, only mask action
            action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
            action_mask = F.pad(action_mask, (1 + input_len, -1), value=True)  # include eos token and input
        action_mask[:, :input_len] = False
        action_mask = action_mask[:, 1:]
        action_mask = action_mask[:, -(sequences.size(1) - input_len) :]
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
