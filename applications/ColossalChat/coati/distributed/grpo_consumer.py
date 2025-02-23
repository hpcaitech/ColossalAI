from contextlib import nullcontext
from typing import Optional

import ray
import torch
from coati.distributed.consumer import BaseConsumer
from coati.distributed.loss import PolicyLoss
from coati.distributed.reward.reward_fn import math_reward_fn
from coati.distributed.reward.verifiable_reward import VerifiableReward
from coati.distributed.utils import calc_action_log_probs
from transformers import AutoModelForCausalLM, AutoTokenizer

from colossalai.nn.optimizer import HybridAdam


@ray.remote
class GRPOConsumer(BaseConsumer):
    def __init__(
        self,
        num_producers,
        num_episodes,
        rank,
        world_size,
        master_addr,
        master_port,
        num_update_per_episode,
        num_recv_per_update,
        batch_size,
        model_config,
        plugin_config,
        microbatch_size=1,
    ):
        super().__init__(
            num_producers,
            num_episodes,
            rank,
            world_size,
            master_addr,
            master_port,
            num_update_per_episode,
            num_recv_per_update,
            batch_size,
            model_config,
            plugin_config,
            microbatch_size,
        )
        path = model_config.pop("path")
        self.policy_model = AutoModelForCausalLM.from_pretrained(path, **model_config)
        self.policy_model.train()
        self.policy_model.gradient_checkpointing_enable()
        self.optimizer = HybridAdam(self.policy_model.parameters(), lr=1e-4)
        self.accum_loss = torch.zeros(1, device=self.device)

        # Reference model is initialized from policy model.
        self.reference_model = AutoModelForCausalLM.from_pretrained(path, **model_config)
        self.reference_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.pad_token_id = self.tokenizer.pad_token_id

        # Initialize verifiable reward.
        response_format_tags = {
            "think_start": {"text": "<think>", "num_occur": 1},
            "think_end": {"text": "</think>", "num_occur": 1},
            "answer_start": {"text": "<answer>", "num_occur": 1},
            "answer_end": {"text": "</answer>", "num_occur": 1},
        }
        self.reward_model = VerifiableReward(
            reward_fns=[math_reward_fn], tokenizer=self.tokenizer, tags=response_format_tags
        )

        self.policy_loss_fn = PolicyLoss()

    def setup(self):
        super().setup()
        self.policy_model, self.optimizer, *_ = self.booster.boost(self.policy_model, self.optimizer)
        self.reference_model, *_ = self.booster.boost(self.reference_model)

    def step(self, step_idx: int, **kwargs) -> Optional[float]:
        """
        Step data from policy model:
            [{
                "input_ids": torch.Tensor,
                "attention_mask": torch.Tensor,
                "action_mask": torch.Tensor,
                "action_log_probs": torch.Tensor,
            },
            ...]
        Format:
            [batch_size, prompt_length + response_length] --- <PAD>...<PAD><PROMPT>...<PROMPT><RESPONSE>...<RESPONSE><PAD>...<PAD>.
        """
        labels = kwargs["input_ids"].clone()
        labels[kwargs["attention_mask"] == 0] = -100
        kwargs["labels"] = labels
        sequences = kwargs["input_ids"]
        action_mask = kwargs["action_mask"]
        num_action = action_mask.shape[1]
        old_action_log_probs = kwargs["action_log_probs"]
        assert kwargs.pop("action_mask").shape == kwargs.pop("action_log_probs").shape

        need_update = (step_idx + 1) % self.num_microbatches == 0

        ctx = nullcontext() if need_update else self.booster.no_sync(self.policy_model, self.optimizer)
        with ctx:
            policy_model_logits = self.policy_model(
                input_ids=kwargs["input_ids"],
                attention_mask=kwargs["attention_mask"],
            )["logits"]
            action_log_probs = calc_action_log_probs(policy_model_logits, sequences, num_action)

            reference_model_logits = self.reference_model(
                input_ids=sequences,
                attention_mask=kwargs["attention_mask"],
            )["logits"]
            reference_action_log_probs = calc_action_log_probs(reference_model_logits, sequences, num_action)

            # GRPO advantage calculation
            kl = torch.sum(-0.01 * (action_log_probs - reference_action_log_probs) * action_mask, dim=-1) / torch.sum(
                action_mask, dim=-1
            )

            reward = self.reward_model(sequences, gt_answer=kwargs["gt_answer"])
            reward = reward + kl
            mean = reward.view(-1, reward.size(0)).mean(dim=1)
            std = reward.view(-1, reward.size(0)).std(dim=1)
            advantages = (reward - mean) / (std + 1e-4)
            # Calculate Loss
            loss, skip_update, _ = self.policy_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages.unsqueeze(dim=-1).repeat_interleave(action_log_probs.size(-1), dim=-1),
                action_mask,
            )

            loss = loss / self.num_microbatches
            self.accum_loss.add_(loss.data)
            if not skip_update:
                self.booster.backward(loss, self.optimizer)
        if need_update:
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_scalar = self.accum_loss.item()
            self.accum_loss.zero_()
            return loss_scalar

    def state_dict(self):
        self.policy_model._force_wait_all_gather()
        model = self.policy_model.unwrap()
        state_dict = model.state_dict()
        return state_dict
