"""
experience maker.
"""

import torch
import torch.nn.functional as F
from coati.models import Critic, RewardModel
from coati.models.generation import generate
from coati.models.utils import calc_action_log_probs, compute_reward
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import Experience, ExperienceMaker


class NaiveExperienceMaker(ExperienceMaker):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: PreTrainedModel,
        critic: Critic,
        reward_model: RewardModel,
        initial_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        kl_coef: float = 0.01,
        gamma: float = 1.0,
        lam: float = 0.95,
    ) -> None:
        super().__init__(actor, critic, reward_model, initial_model)
        self.tokenizer = tokenizer
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.lam = lam

    @torch.no_grad()
    def calculate_advantage(self, value, reward, num_actions):
        lastgaelam = 0
        advantages_reversed = []
        for t in reversed(range(num_actions)):
            nextvalues = value[:, t + 1] if t < num_actions - 1 else 0.0
            delta = reward[:, t] + self.gamma * nextvalues - value[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        return advantages

    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()
        torch.manual_seed(47)  # for tp, gurantee the same input for reward model
        sequences = generate(self.actor, input_ids, self.tokenizer, **generate_kwargs)
        sequence_length = sequences.size(1)

        # calculate auxiliary tensors
        attention_mask = None
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            attention_mask = sequences.not_equal(pad_token_id).to(dtype=torch.long, device=sequences.device)

        input_len = input_ids.size(1)
        eos_token_id = self.tokenizer.eos_token_id
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

        actor_output = self.actor(input_ids=sequences, attention_mask=attention_mask)["logits"]
        action_log_probs = calc_action_log_probs(actor_output, sequences, num_actions)

        base_model_output = self.initial_model(input_ids=sequences, attention_mask=attention_mask)["logits"]

        base_action_log_probs = calc_action_log_probs(base_model_output, sequences, num_actions)
        value = self.critic(input_ids=sequences, attention_mask=attention_mask)

        # convert from left padding to right padding
        input_ids_rm = torch.zeros_like(sequences, device=sequences.device)
        attention_mask_rm = torch.zeros_like(sequences, device=sequences.device)
        for i in range(sequences.size(0)):
            sequence = sequences[i]
            bos_index = (sequence == self.tokenizer.bos_token_id).nonzero().squeeze()[0]
            eos_index = int(
                (torch.arange(sequence_length, device=sequence.device) * (sequence != self.tokenizer.pad_token_id))
                .max()
                .item()
            )
            sequence_to_pad = sequence[bos_index : eos_index + 1]
            sequence_padded = F.pad(
                sequence_to_pad, (0, sequence_length - sequence_to_pad.size(0)), value=self.tokenizer.pad_token_id
            )
            input_ids_rm[i] = sequence_padded
            if sequence_length - sequence_to_pad.size(0) > 0:
                attention_mask_rm[i, : sequence_to_pad.size(0) + 1] = 1
            else:
                attention_mask_rm[i, :] = 1
        attention_mask_rm = attention_mask_rm.to(dtype=torch.bool)
        torch.set_printoptions(threshold=10_000)

        r = self.reward_model(
            input_ids=input_ids_rm.to(dtype=torch.long, device=sequences.device),
            attention_mask=attention_mask_rm.to(device=sequences.device),
        )

        reward, kl = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask)
        value = value[:, -num_actions:] * action_mask
        advantages = self.calculate_advantage(value, reward, num_actions)

        advantages = advantages.detach()
        value = value.detach()
        r = r.detach()

        return Experience(sequences, action_log_probs, value, r, kl, advantages, attention_mask, action_mask)
