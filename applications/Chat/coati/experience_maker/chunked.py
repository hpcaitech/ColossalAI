from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from coati.models.base import Actor, Critic, RewardModel
from coati.models.generation import generate
from coati.models.utils import calc_action_log_probs
from transformers import PreTrainedTokenizer

from .base import Experience, ExperienceMaker


class ChunkedExperienceMaker(ExperienceMaker):
    """
    Chunked experience maker.
    NOTE: Treat every `chunk_size` tokens chunk as a step in MDP.
        chunk_size = 1: every token is a step
        chunk_size = seq_len: all the tokens constitute a giant step
    """

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        reward_model: RewardModel,
        initial_model: Actor,
        tokenizer: PreTrainedTokenizer,
        chunk_size: int = 8,
        kl_coef: float = 0.1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        super().__init__(actor, critic, reward_model, initial_model)
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    @staticmethod
    @torch.no_grad()
    def compute_advantages_and_returns(
        values: torch.Tensor,
        rewards: torch.Tensor,
        end_flags: torch.Tensor,
        num_steps: int,
        gamma: float,
        gae_lambda: float,
        bootstrap_value: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is modified from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py.

        Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Args:
            values: Tensor of shape (batch_size, num_steps + 1),
                V(s_0), V(s_1), ..., V(s_N), V(s_N+1)
            rewards: Tensor of shape (batch_size, num_steps),
                r(s_0, a_0), r(s_1, a_1), ..., r(s_N, a_N)
            end_flags: Tensor of shape (batch_size, num_steps),
                NOTE: end_flags[i] indicates whether the s_{i+1} is the terminal state.
            num_steps: Length of MDP
            gamma: γ, discount factor
            gae_lambda: λ, GAE parameter
            bootstrap_value: Whether to bootstrap the terminal value or not.
                i.e., V(s_T+1) = 0 if bootstrap_value is False, otherwise V(s_T+1) = Critic(s_T+1)
                This may be useful when the MDP is truncated by the max length.
        """

        last_gae_lambda = 0
        advantages = torch.zeros_like(rewards)
        for t in reversed(range(num_steps)):
            next_non_terminal = torch.logical_not(end_flags[:, t])
            next_values = values[:, t + 1] * (next_non_terminal + bootstrap_value).bool()
            delta = rewards[:, t] + gamma * next_values - values[:, t]
            last_gae_lambda = delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
            advantages[:, t] = last_gae_lambda
        returns = advantages + values[:, :-1]
        return advantages, returns

    @torch.no_grad()
    def make_experience(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **generate_kwargs
    ) -> Tuple[Experience, Dict]:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        # generate sequences
        sequences = generate(
            self.actor, input_ids, tokenizer=self.tokenizer, attention_mask=attention_mask, **generate_kwargs
        )

        # calculate auxiliary tensors
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        assert (
            eos_token_id is not None and pad_token_id is not None
        ), "eos_token_id and pad_token_id must be specified in generate_kwargs"
        input_len = input_ids.size(1)
        num_actions = sequences.size(1) - input_len
        num_steps = (num_actions + self.chunk_size - 1) // self.chunk_size

        # if action is |action|eos|pad|, then action_mask is |1|1|0|
        action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
        action_mask = F.pad(action_mask, (1, -1), value=True)  # shift right by 1 to include eos token
        step_mask = F.pad(action_mask, (0, (self.chunk_size - num_actions) % self.chunk_size), value=False).view(
            -1, self.chunk_size
        )
        step_mask = (step_mask.sum(dim=-1) > 0).view(-1, num_steps)
        attention_mask = torch.cat([attention_mask, action_mask], dim=-1)

        # compute action log probs
        actor_logits = self.actor(sequences, attention_mask)["logits"]
        action_log_probs = calc_action_log_probs(actor_logits, sequences, num_actions)
        base_model_logits = self.initial_model(sequences, attention_mask)["logits"]
        base_log_probs = calc_action_log_probs(base_model_logits, sequences, num_actions)

        log_ratio = action_log_probs - base_log_probs
        log_ratio = F.pad(log_ratio, (0, (self.chunk_size - num_actions) % self.chunk_size), value=0).view(
            -1, self.chunk_size
        )
        log_ratio_mask = F.pad(action_mask, (0, (self.chunk_size - num_actions) % self.chunk_size), value=False).view(
            -1, self.chunk_size
        )
        chunk_log_ratio = torch.sum(log_ratio * log_ratio_mask, dim=-1).view(-1, num_steps)

        # compute V(s_i)
        values = torch.zeros((sequences.size(0), num_steps + 1), device=sequences.device)
        # TODO(cwher): employ kv cache?
        # TODO(cwher): is it necessary to add <eos>?
        for i in range(num_steps + 1):
            seq_len = input_len + min(i * self.chunk_size, num_actions)
            sequence_with_eos = F.pad(sequences[:, :seq_len], (0, 1), value=eos_token_id)
            sequence_with_eos_mask = F.pad(attention_mask[:, :seq_len], (0, 1), value=False)
            # NOTE: sequences[:, :seq_len] must contain <eos> if sequences[:, seq_len - 1] is {<eos>, padding token}
            sequence_with_eos_mask[:, -1] = torch.logical_and(
                sequences[:, seq_len - 1] != pad_token_id, sequences[:, seq_len - 1] != eos_token_id
            )
            values[:, i] = self.critic(sequence_with_eos, sequence_with_eos_mask)
        final_rewards = self.reward_model(sequence_with_eos, sequence_with_eos_mask)

        # NOTE: reward is calculated according to the following rules:
        #   1. reward[i] = -kl_coef * chunk_log_ratio[i]
        #   2. reward[-1] += final_reward
        rewards = -self.kl_coef * chunk_log_ratio * step_mask
        # NOTE: actions may contain padding tokens
        num_valid_actions = action_mask.sum(dim=-1)
        num_valid_steps = (num_valid_actions + self.chunk_size - 1) // self.chunk_size
        rewards[torch.arange(rewards.size(0)), num_valid_steps - 1] += final_rewards

        end_flags = torch.zeros_like(rewards, dtype=torch.bool)
        end_flags[torch.arange(rewards.size(0)), num_valid_steps - 1] = True

        advantages, returns = self.compute_advantages_and_returns(
            values, rewards, end_flags, num_steps, self.gamma, self.gae_lambda
        )

        experience = Experience(
            sequences, attention_mask, action_mask, step_mask, action_log_probs, values[:, :-1], returns, advantages
        )

        metrics = {
            "episode_rewards": rewards.sum(dim=-1).mean().item(),
            "final_rewards": final_rewards.mean().item(),
            "episode_steps": num_valid_steps.float().mean().item(),
            "episode_tokens": num_valid_actions.float().mean().item(),
        }

        return experience, metrics
