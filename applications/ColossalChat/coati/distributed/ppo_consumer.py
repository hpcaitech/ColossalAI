from contextlib import nullcontext
from typing import Optional

import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from coati.distributed.consumer import BaseConsumer
from coati.distributed.loss import PolicyLoss, ValueLoss
from coati.distributed.reward.reward_fn import math_reward_fn
from coati.distributed.reward.verifiable_reward import VerifiableReward
from coati.distributed.utils import calc_action_log_probs, compute_reward_ppo
from coati.models import Critic, disable_dropout
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

from colossalai.nn.optimizer import HybridAdam


@ray.remote
class PPOConsumer(BaseConsumer):
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
        rm_path=None,
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
        rm_path = model_config.pop("rm_path")
        self.input_len = model_config.pop("input_len")
        self.kl_coef = model_config.pop("kl_coef")
        self.writer = SummaryWriter(log_dir="runs/experiment_1")
        self.gamma = model_config.pop("gamma")
        self.lamda = model_config.pop("lamda")
        self.policy_model = AutoModelForCausalLM.from_pretrained(path, **model_config)
        self.policy_model.train()
        self.policy_model.gradient_checkpointing_enable()
        self.critic_model = Critic(rm_path, **model_config)
        self.critic_model.model.gradient_checkpointing_enable()
        self.critic_model.train()
        # Disable dropout
        disable_dropout(self.policy_model)
        disable_dropout(self.critic_model)
        self.policy_optimizer = HybridAdam(self.policy_model.parameters(), lr=1e-6)
        self.critic_optimizer = HybridAdam(self.critic_model.parameters(), lr=5e-6)
        self.accum_loss = torch.zeros(1, device=self.device)
        self.accum_kl = torch.zeros(1, device=self.device)
        self.accum_reward = torch.zeros(1, device=self.device)
        self.accum_max_reward = torch.zeros(1, device=self.device)
        self.accum_advantages = torch.zeros(1, device=self.device)
        self.accum_value = torch.zeros(1, device=self.device)
        self.accum_critic_loss = torch.zeros(1, device=self.device)
        self.accum_micro_batch = 0
        self.global_step = 0

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
        self.critic_loss_fn = ValueLoss()

    def setup(self):
        super().setup()
        self.policy_model, self.policy_optimizer, *_ = self.booster.boost(self.policy_model, self.policy_optimizer)
        self.critic_model, self.critic_optimizer, *_ = self.critic_booster.boost(
            self.critic_model, self.critic_optimizer
        )
        self.reference_model, *_ = self.critic_booster.boost(self.reference_model)

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
        sequence_length = sequences.size(1)
        num_actions = sequence_length - self.input_len
        action_mask = kwargs["action_mask"]
        generation_end_index = (action_mask != 0).cumsum(dim=1).argmax(dim=1) + self.input_len
        num_action = action_mask.shape[1]
        old_action_log_probs = kwargs["action_log_probs"]
        assert kwargs.pop("action_mask").shape == kwargs.pop("action_log_probs").shape

        need_update = (step_idx + 1) % self.num_microbatches == 0

        ctx = nullcontext()  # if need_update else self.booster.no_sync(self.policy_model, self.policy_optimizer)
        with ctx:
            policy_model_logits = self.policy_model(
                input_ids=sequences,
                attention_mask=kwargs["attention_mask"],
            )["logits"]
            action_log_probs = calc_action_log_probs(policy_model_logits, sequences, num_action)
            reference_model_logits = self.reference_model(
                input_ids=sequences,
                attention_mask=kwargs["attention_mask"],
            )["logits"]
            reference_action_log_probs = calc_action_log_probs(reference_model_logits, sequences, num_action)

            # Convert to right padding for the reward model and the critic model
            input_ids_rm = torch.zeros_like(sequences, device=sequences.device)
            response_start = []
            response_end = []
            attention_mask_rm = torch.zeros_like(sequences, device=sequences.device)
            for i in range(sequences.size(0)):
                sequence = sequences[i]
                bos_index = (sequence != self.pad_token_id).nonzero().reshape([-1])[0]
                eos_index = generation_end_index[i] + 1  # include the stop token
                sequence_to_pad = sequence[bos_index:eos_index]
                response_start.append(self.input_len - bos_index)
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
            # print(response_start, response_end,generation_end_index)
            r = self.reward_model(
                input_ids_rm, gt_answer=kwargs["gt_answer"], response_start=response_start, response_end=response_end
            )
            value = self.critic_model(
                input_ids=sequences,
                attention_mask=attention_mask_rm.to(device=sequences.device),
            )
            value = value[:, -num_actions:] * action_mask
            reward, kl = compute_reward_ppo(
                r, self.kl_coef, action_log_probs, reference_action_log_probs, action_mask=action_mask
            )

            # Advantage Calculation
            lastgaelam = 0
            advantages_reversed = []
            for t in reversed(range(num_actions)):
                nextvalues = value[:, t + 1] if t < num_actions - 1 else 0.0
                delta = reward[:, t] + self.gamma * nextvalues - value[:, t]
                lastgaelam = delta + self.gamma * self.lamda * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)
            advantages = advantages.detach()

            # Policy Loss
            # print(f"debug: action_log_probs:{action_log_probs.size()}, old_action_log_probs:{old_action_log_probs.size()}, advantages:{advantages.size()}, action_mask:{action_mask.size()}")
            loss, skip_update, _ = self.policy_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages,
                action_mask,
            )

            # Critic Loss
            # Hack: use the current value to approximate the old value, should be old value mathematically
            critic_loss = self.critic_loss_fn(
                value,
                value.detach(),
                advantages,
                action_mask=action_mask,
            )

            loss = loss / self.num_microbatches
            self.accum_loss.add_(loss.data)
            self.accum_kl.add_(kl.mean().data)
            self.accum_reward.add_(r.mean().data)
            self.accum_max_reward.add_(r.max().data)
            self.accum_advantages.add_(advantages.mean().data)
            self.accum_value.add_(value.mean().data)
            self.accum_critic_loss.add_(critic_loss.mean().data)
            self.accum_micro_batch += 1

            if not skip_update:
                if self.global_step > 1:
                    self.booster.backward(loss, self.policy_optimizer)
                self.critic_booster.backward(critic_loss, self.critic_optimizer)
        if need_update:
            if self.global_step > 1:
                self.policy_optimizer.step()
                self.policy_optimizer.zero_grad()
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()
            loss_scalar = self.accum_loss.item()
            self.accum_loss.zero_()
            mean_kl = self.accum_kl.item() / self.accum_micro_batch
            mean_reward = self.accum_reward.item() / self.accum_micro_batch
            if dist.get_rank() == 0:
                # TODO: add all gather
                self.writer.add_scalar("Loss", loss_scalar, self.global_step)
                self.writer.add_scalar("KL", mean_kl, self.global_step)
                self.writer.add_scalar("Reward", mean_reward, self.global_step)
                self.writer.add_scalar(
                    "Max Reward", self.accum_max_reward.item() / self.accum_micro_batch, self.global_step
                )
                self.writer.add_scalar(
                    "Advantages", self.accum_advantages.item() / self.accum_micro_batch, self.global_step
                )
                self.writer.add_scalar("Value", self.accum_value.item() / self.accum_micro_batch, self.global_step)
                self.writer.add_scalar(
                    "Critic Loss", self.accum_critic_loss.item() / self.accum_micro_batch, self.global_step
                )
                if self.global_step % 1 == 0:
                    self.writer.add_text(
                        "Generation",
                        self.tokenizer.decode(sequences[0], skip_special_tokens=True),
                        self.global_step,
                    )
            self.accum_micro_batch = 0
            self.accum_reward.zero_()
            self.accum_max_reward.zero_()
            self.accum_kl.zero_()
            self.accum_advantages.zero_()
            self.accum_value.zero_()
            self.accum_critic_loss.zero_()
            self.global_step += 1
            return loss_scalar

    def state_dict(self):
        self.policy_model._force_wait_all_gather()
        model = self.policy_model.unwrap()
        state_dict = model.state_dict()
        return state_dict
