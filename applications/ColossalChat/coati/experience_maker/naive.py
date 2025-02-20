"""
experience maker.
"""

from typing import Any

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
        use_grpo: bool = False,
        num_generation: int = 8,
        inference_batch_size: int = None,
        logits_forward_batch_size: int = 2,
    ) -> None:
        super().__init__(actor, critic, reward_model, initial_model)
        self.tokenizer = tokenizer
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.lam = lam
        self.use_grpo = use_grpo
        self.num_generation = num_generation
        self.inference_batch_size = inference_batch_size
        self.logits_forward_batch_size = logits_forward_batch_size
        if not self.use_grpo:
            assert self.critic is not None, "Critic model is required for PPO training."
        else:
            assert self.critic is None, "Critic model is not required for GRPO training."
            assert self.num_generation > 1, "Number of generations should be greater than 1 for GRPO training."

    @torch.inference_mode()
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
    def make_experience(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, gt_answer: Any = None, **generate_kwargs
    ) -> Experience:
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
        if self.critic:
            self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()
        pad_token_id = self.tokenizer.pad_token_id
        stop_token_ids = generate_kwargs.get("stop_token_ids", None)
        if isinstance(stop_token_ids, int):
            stop_token_ids = [[stop_token_ids]]
        elif isinstance(stop_token_ids[0], int):
            stop_token_ids = [stop_token_ids]
        elif isinstance(stop_token_ids[0], list):
            pass
        else:
            raise ValueError(
                f"stop_token_ids should be a list of list of integers, a list of integers or an integers. got {stop_token_ids}"
            )
        generate_kwargs["stop_token_ids"] = stop_token_ids
        torch.manual_seed(41)  # for tp, gurantee the same input for reward model

        if self.use_grpo and self.num_generation > 1:
            # Generate multiple responses for each prompt
            input_ids = input_ids.repeat_interleave(self.num_generation, dim=0)
            gt_answer_tmp = []
            for t in gt_answer:
                gt_answer_tmp.extend([t] * self.num_generation)
            gt_answer = gt_answer_tmp
        if self.inference_batch_size is None:
            self.inference_batch_size = input_ids.size(0)

        batch_sequences = []
        batch_input_ids_rm = []
        batch_attention_mask_rm = []
        batch_attention_mask = []
        batch_r = []
        batch_action_log_probs = []
        batch_base_action_log_probs = []
        batch_action_mask = []
        num_actions = 0

        for inference_mini_batch_id in range(0, input_ids.size(0), self.inference_batch_size):
            s, e = inference_mini_batch_id, inference_mini_batch_id + self.inference_batch_size
            if input_ids[s:e].size(0) == 0:
                break
            sequences = generate(self.actor, input_ids[s:e], self.tokenizer, **generate_kwargs)
            # pad to max_len, you don't want to get an OOM error after a thousands of steps
            sequences = F.pad(sequences, (0, generate_kwargs["max_length"] - sequences.size(1)), value=pad_token_id)

            # Pad to max length
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
                    stop_token_pos = [
                        find_first_occurrence_subsequence(
                            sequences[i][input_len:], torch.tensor(stop_token_id).to(sequences.device)
                        )
                        for stop_token_id in stop_token_ids
                    ]
                    stop_index = min([i for i in stop_token_pos if i != -1], default=-1)
                    stop_token_id = stop_token_ids[stop_token_pos.index(stop_index)]
                    if stop_index == -1:
                        # Sequence does not contain stop_token_ids, this should never happen BTW
                        logger.warning(
                            "Generated sequence does not contain stop_token_ids. Please check your chat template config"
                        )
                        print(self.tokenizer.decode(sequences[i], skip_special_tokens=True))
                    else:
                        # Keep stop tokens
                        stop_index = input_len + stop_index
                        action_mask[i, stop_index + len(stop_token_id) :] = False

            generation_end_index = (action_mask == True).sum(dim=-1) - 1
            action_mask[:, :input_len] = False
            action_mask = action_mask[:, 1:]
            action_mask = action_mask[:, -(sequences.size(1) - input_len) :]
            num_actions = action_mask.size(1)
            torch.cuda.empty_cache()
            with torch.inference_mode():
                actor_output = []
                base_model_output = []
                for i in range(0, sequences.size(0), self.logits_forward_batch_size):
                    actor_output.append(
                        self.actor(
                            input_ids=sequences[i : i + self.logits_forward_batch_size],
                            attention_mask=attention_mask[i : i + self.logits_forward_batch_size],
                            use_cache=False,
                        )["logits"]
                    )
                    base_model_output.append(
                        self.initial_model(
                            input_ids=sequences[i : i + self.logits_forward_batch_size],
                            attention_mask=attention_mask[i : i + self.logits_forward_batch_size],
                            use_cache=False,
                        )["logits"]
                    )
                actor_output = torch.cat(actor_output, dim=0)
                base_model_output = torch.cat(base_model_output, dim=0)
                action_log_probs = calc_action_log_probs(actor_output, sequences, num_actions)
                base_action_log_probs = calc_action_log_probs(base_model_output, sequences, num_actions)

            # Convert to right padding for the reward model and the critic model
            input_ids_rm = torch.zeros_like(sequences, device=sequences.device)
            response_start = []
            response_end = []
            attention_mask_rm = torch.zeros_like(sequences, device=sequences.device)
            for i in range(sequences.size(0)):
                sequence = sequences[i]
                bos_index = (sequence != pad_token_id).nonzero().reshape([-1])[0]
                eos_index = generation_end_index[i] + 1  # include the stop token
                sequence_to_pad = sequence[bos_index:eos_index]
                response_start.append(input_len - bos_index)
                response_end.append(eos_index - bos_index)
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
                response_start=response_start,
                response_end=response_end,
                gt_answer=gt_answer[s:e],
            )

            batch_sequences.append(sequences)
            batch_input_ids_rm.append(input_ids_rm)
            batch_attention_mask_rm.append(attention_mask_rm)
            batch_attention_mask.append(attention_mask)
            batch_r.append(r)
            batch_action_log_probs.append(action_log_probs.cpu())
            batch_base_action_log_probs.append(base_action_log_probs.cpu())
            batch_action_mask.append(action_mask)

        sequences = torch.cat(batch_sequences, dim=0)
        input_ids_rm = torch.cat(batch_input_ids_rm, dim=0)
        attention_mask_rm = torch.cat(batch_attention_mask_rm, dim=0)
        attention_mask = torch.cat(batch_attention_mask, dim=0)
        r = torch.cat(batch_r, dim=0)
        action_log_probs = torch.cat(batch_action_log_probs, dim=0).to(sequences.device)
        base_action_log_probs = torch.cat(batch_base_action_log_probs, dim=0).to(sequences.device)
        action_mask = torch.cat(batch_action_mask, dim=0).to(sequences.device)
        if not self.use_grpo:
            value = self.critic(
                input_ids=input_ids_rm.to(dtype=torch.long, device=sequences.device),
                attention_mask=attention_mask_rm.to(device=sequences.device),
            )
            value = value[:, -num_actions:] * action_mask
            reward, kl = compute_reward(
                r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask
            )
            advantages = self.calculate_advantage(value, reward, num_actions)
            advantages = advantages.detach()
            value = value.detach()
        else:
            # GRPO advantage calculation
            kl = torch.sum(
                -self.kl_coef * (action_log_probs - base_action_log_probs) * action_mask, dim=-1
            ) / torch.sum(
                action_mask, dim=-1
            )  # address numerical instability issue
            r = kl + r
            mean_gr = r.view(-1, self.num_generation).mean(dim=1)
            std_gr = r.view(-1, self.num_generation).std(dim=1)
            mean_gr = mean_gr.repeat_interleave(self.num_generation, dim=0)
            std_gr = std_gr.repeat_interleave(self.num_generation, dim=0)
            advantages = (r - mean_gr) / (std_gr + 1e-4)
            value = r.detach()  # dummy value
        r = r.detach()
        return Experience(
            sequences.cpu(),
            action_log_probs.cpu(),
            value.cpu(),
            r.cpu(),
            kl.cpu(),
            advantages.cpu(),
            attention_mask.cpu(),
            action_mask.cpu(),
        )
