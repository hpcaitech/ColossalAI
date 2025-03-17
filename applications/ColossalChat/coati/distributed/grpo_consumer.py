from contextlib import nullcontext
from typing import Optional

import ray
import torch
import wandb
from coati.distributed.consumer import BaseConsumer
from coati.distributed.loss import PolicyLoss
from coati.distributed.reward.reward_fn import math_reward_fn
from coati.distributed.reward.verifiable_reward import VerifiableReward
from coati.distributed.utils import calc_action_log_probs
from coati.trainer.utils import all_reduce_mean
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
        num_generations=4,
        use_wandb=True,
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
        self.optimizer = HybridAdam(self.policy_model.parameters(), lr=1e-6)
        self.accum_loss = torch.zeros(1, device=self.device)
        self.accum_reward = torch.zeros(1, device=self.device)
        self.accum_kl = torch.zeros(1, device=self.device)
        self.accum_format_reward = torch.zeros(1, device=self.device)
        self.accum_acc_reward = torch.zeros(1, device=self.device)
        self.accum_advantages = torch.zeros(1, device=self.device)
        self.accum_response_length = torch.zeros(1, device=self.device)
        self.accum_count = 0

        # Reference model is initialized from policy model.
        self.reference_model = AutoModelForCausalLM.from_pretrained(path, **model_config)
        self.reference_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.num_generations = num_generations

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
        self.global_step = 0
        if use_wandb and self.rank == 0:
            self.wandb_run = wandb.init(project="GRPO-V1", sync_tensorboard=True)

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
            [batch_size, num_of_generation, prompt_length + response_length] --- <PAD>...<PAD><PROMPT>...<PROMPT><RESPONSE>...<RESPONSE><PAD>...<PAD>.
        """

        # Reshape to [batch_size x num_of_generation, prompt_length + response_length]
        data = {k: v.view(-1, v.size(-1)) for k, v in kwargs.items()}
        action_mask = data["action_mask"]
        num_action = action_mask.shape[1]
        old_action_log_probs = data["action_log_probs"]
        response_length = torch.sum(action_mask, dim=1).to(torch.float32)

        need_update = (step_idx + 1) % self.num_microbatches == 0

        ctx = nullcontext() if need_update else self.booster.no_sync(self.policy_model, self.optimizer)
        with ctx:
            policy_model_logits = self.policy_model(
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
            )["logits"]
            action_log_probs = calc_action_log_probs(
                policy_model_logits, data["input_ids"], num_action, self.plugin.shard_config
            )

            with torch.no_grad():
                reference_model_logits = self.reference_model(
                    input_ids=data["input_ids"],
                    attention_mask=data["attention_mask"],
                )["logits"]
            reference_action_log_probs = calc_action_log_probs(
                reference_model_logits, data["input_ids"], num_action, self.plugin.shard_config
            )

            per_token_kl = (
                torch.exp(reference_action_log_probs - action_log_probs)
                - (reference_action_log_probs - action_log_probs)
                - 1
            )
            kl = torch.sum(per_token_kl * action_mask, dim=-1) / torch.sum(action_mask, dim=-1)

            reward_group = self.reward_model(
                data["input_ids"], gt_answer=data["gt_answer"], response_idx=data["response_idx"]
            )

            reward = torch.tensor([value[0] for value in reward_group]).to(data["input_ids"].device)
            format_reward = torch.tensor([value[1] for value in reward_group]).to(data["input_ids"].device)
            acc_reward = torch.tensor([value[2] for value in reward_group]).to(data["input_ids"].device)

            # [batch_size, num_generations]
            group_reward = reward.view(-1, self.num_generations)

            # [batch_size x num_generations]
            reward_mean = group_reward.mean(dim=1).repeat_interleave(self.num_generations, dim=0)
            reward_std = group_reward.std(dim=1).repeat_interleave(self.num_generations, dim=0)
            # [batch_size x num_generations]
            advantages = (reward - reward_mean) / (reward_std + 1e-4)

            # Calculate Loss
            loss, skip_update, _ = self.policy_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages.unsqueeze(dim=-1).repeat_interleave(action_log_probs.size(-1), dim=-1),
                per_token_kl,
                action_mask,
            )

            if not skip_update:
                self.booster.backward(loss, self.optimizer)
            loss = all_reduce_mean(loss, self.plugin)
            reward = all_reduce_mean(reward.mean(), self.plugin)
            kl = all_reduce_mean(kl.mean(), self.plugin)
            format_reward = all_reduce_mean(format_reward.mean(), self.plugin)
            acc_reward = all_reduce_mean(acc_reward.mean(), self.plugin)
            advantages = all_reduce_mean(advantages.mean(), self.plugin)
            response_length = all_reduce_mean(response_length.mean(), self.plugin)
            # Calculate accumulate value.
            self.accum_loss.add_(loss.data)
            self.accum_reward.add_(reward.data)
            self.accum_kl.add_(kl.data)
            self.accum_format_reward.add_(format_reward.data)
            self.accum_acc_reward.add_(acc_reward.data)
            self.accum_advantages.add_(advantages.data)
            self.accum_response_length.add_(response_length.data)
            self.accum_count += 1
        if need_update:
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_scalar = self.accum_loss.item()
            if self.rank == 0:
                print(
                    "Loss:",
                    self.accum_loss.item() / self.accum_count,
                    "\nReward:",
                    self.accum_reward.item() / self.accum_count,
                    "\nFormat Reward:",
                    self.accum_format_reward.item() / self.accum_count,
                    "\nAcc Reward:",
                    self.accum_acc_reward.item() / self.accum_count,
                    "\nKL:",
                    self.accum_kl.item() / self.accum_count,
                    "\nAdvantages:",
                    self.accum_advantages.item() / self.accum_count,
                    "\nResponse Length:",
                    self.accum_response_length.item() / self.accum_count,
                )
                self.wandb_run.log(
                    {
                        "train/loss": self.accum_loss.item() / self.accum_count,
                        "train/reward": self.accum_reward.item() / self.accum_count,
                        "train/format_reward": self.accum_format_reward.item() / self.accum_count,
                        "train/acc_reward": self.accum_acc_reward.item() / self.accum_count,
                        "train/kl": self.accum_kl.item() / self.accum_count,
                        "train/advantages": self.accum_advantages.item() / self.accum_count,
                        "train/response_length": self.accum_response_length.item() / self.accum_count,
                    }
                )
            self.accum_loss.zero_()
            self.accum_reward.zero_()
            self.accum_acc_reward.zero_()
            self.accum_format_reward.zero_()
            self.accum_kl.zero_()
            self.accum_advantages.zero_()
            self.accum_response_length.zero_()

            self.accum_count = 0
            return loss_scalar

    def state_dict(self):
        self.policy_model._force_wait_all_gather()
        model = self.policy_model.unwrap()
        state_dict = model.state_dict()
        return state_dict
