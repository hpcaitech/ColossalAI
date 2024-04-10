"""
experience maker.
"""

import torch
import torch.nn.functional as F
from coati.dataset.utils import find_first_occurrence_subsequence
from coati.models import Critic, RewardModel
from coati.models.generation import generate
from coati.models.utils import calc_action_log_probs, compute_reward
from transformers import PreTrainedModel, PreTrainedTokenizer

from colossalai.logging import get_dist_logger

from .base import Experience, ExperienceMaker

logger = get_dist_logger()

import torch.distributed as dist


def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


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
    def calculate_advantage(self, value: torch.Tensor, reward: torch.Tensor, num_actions: int) -> torch.Tensor:
        """
        Calculates the advantage values for each action based on the value and reward tensors.

        Args:
            value (torch.Tensor): Tensor containing the predicted values from critic.
            reward (torch.Tensor): reward of the shape [B, len].
            num_actions (int): Number of actions.

        Returns:
            torch.Tensor: Tensor containing the calculated advantages for each action.
        """
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
        """
        Generates an experience using the given input_ids and attention_mask.

        Args:
            input_ids (torch.Tensor): The input tensor containing the tokenized input sequence.
            attention_mask (torch.Tensor): The attention mask tensor indicating which tokens to attend to.
            **generate_kwargs: Additional keyword arguments for the generation process.

        Returns:
            Experience: The generated experience object.

        """
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()
        pad_token_id = self.tokenizer.pad_token_id

        stop_token_ids = generate_kwargs.get("stop_token_ids", None)
        torch.manual_seed(41)  # for tp, gurantee the same input for reward model

        sequences = generate(self.actor, input_ids, self.tokenizer, **generate_kwargs)

        # Pad to max length
        sequences = F.pad(sequences, (0, generate_kwargs["max_length"] - sequences.size(1)), value=pad_token_id)
        sequence_length = sequences.size(1)

        # Calculate auxiliary tensors
        attention_mask = None
        if pad_token_id is not None:
            attention_mask = sequences.not_equal(pad_token_id).to(dtype=torch.long, device=sequences.device)

        input_len = input_ids.size(1)
        if stop_token_ids is None:
            # End the sequence with eos token
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is None:
                action_mask = torch.ones_like(sequences, dtype=torch.bool)
            else:
                # Left padding may be applied, only mask action
                action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
                action_mask = F.pad(action_mask, (1 + input_len, -1), value=True)  # include eos token and input
        else:
            # stop_token_ids are given, generation ends with stop_token_ids
            action_mask = torch.ones_like(sequences, dtype=torch.bool)
            for i in range(sequences.size(0)):
                stop_index = find_first_occurrence_subsequence(
                    sequences[i][input_len:], torch.tensor(stop_token_ids).to(sequences.device)
                )
                if stop_index == -1:
                    # Sequence does not contain stop_token_ids, this should never happen BTW
                    logger.warning(
                        "Generated sequence does not contain stop_token_ids. Please check your chat template config"
                    )
                else:
                    # Keep stop tokens
                    stop_index = input_len + stop_index
                    action_mask[i, stop_index + len(stop_token_ids) :] = False

        generation_end_index = (action_mask == True).sum(dim=-1) - 1
        action_mask[:, :input_len] = False
        action_mask = action_mask[:, 1:]
        action_mask = action_mask[:, -(sequences.size(1) - input_len) :]
        num_actions = action_mask.size(1)

        actor_output = self.actor(input_ids=sequences, attention_mask=attention_mask)["logits"]
        action_log_probs = calc_action_log_probs(actor_output, sequences, num_actions)

        base_model_output = self.initial_model(input_ids=sequences, attention_mask=attention_mask)["logits"]

        base_action_log_probs = calc_action_log_probs(base_model_output, sequences, num_actions)

        # Convert to right padding for the reward model and the critic model
        input_ids_rm = torch.zeros_like(sequences, device=sequences.device)
        attention_mask_rm = torch.zeros_like(sequences, device=sequences.device)
        for i in range(sequences.size(0)):
            sequence = sequences[i]
            bos_index = (sequence != pad_token_id).nonzero().reshape([-1])[0]
            eos_index = generation_end_index[i]
            sequence_to_pad = sequence[bos_index:eos_index]
            sequence_padded = F.pad(
                sequence_to_pad, (0, sequence_length - sequence_to_pad.size(0)), value=self.tokenizer.pad_token_id
            )
            input_ids_rm[i] = sequence_padded
            if sequence_length - sequence_to_pad.size(0) > 0:
                attention_mask_rm[i, : sequence_to_pad.size(0) + 1] = 1
            else:
                attention_mask_rm[i, :] = 1
        attention_mask_rm = attention_mask_rm.to(dtype=torch.bool)

        r = self.reward_model(
            input_ids=input_ids_rm.to(dtype=torch.long, device=sequences.device),
            attention_mask=attention_mask_rm.to(device=sequences.device),
        )

        value = self.critic(
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
