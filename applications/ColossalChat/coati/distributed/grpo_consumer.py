import json
import os
from contextlib import nullcontext
from typing import Optional

import ray
import torch
import torch.distributed as dist
import wandb
from coati.distributed.consumer import BaseConsumer
from coati.distributed.loss import PolicyLoss
from coati.distributed.reward.reward_fn import math_reward_fn
from coati.distributed.reward.verifiable_reward import VerifiableReward
from coati.distributed.utils import calc_action_log_probs, get_logits_rebatched_forward
from coati.trainer.utils import all_reduce_mean
from transformers import AutoModelForCausalLM, AutoTokenizer

from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
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
        num_generations=8,
        use_wandb=True,
        generate_config=None,
        training_config={},
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
        self.optimizer = HybridAdam(self.policy_model.parameters(), lr=training_config.get("lr", 1e-6))
        self.accum_loss = torch.zeros(1, device=self.device)
        self.accum_reward = torch.zeros(1, device=self.device)
        self.accum_kl = torch.zeros(1, device=self.device)
        self.accum_format_reward = torch.zeros(1, device=self.device)
        self.accum_acc_reward = torch.zeros(1, device=self.device)
        self.accum_advantages = torch.zeros(1, device=self.device)
        self.accum_response_length = torch.zeros(1, device=self.device)
        self.accum_count = 0
        self.generate_config = generate_config
        self.training_config = training_config

        # Reference model is initialized from policy model.
        self.reference_model = AutoModelForCausalLM.from_pretrained(path, **model_config)
        self.reference_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.num_generations = num_generations
        self.filter_range = training_config.get("filter_range", None)
        if self.filter_range is not None:
            assert len(self.filter_range) == 2, "Filter range should have 2 values."

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
            name = f"{generate_config['backend']}_bs_{self.batch_size*self.world_size}_temp_{generate_config['temperature']:.01f}_top_p_{generate_config['top_p']:.02f}"
            self.wandb_run = wandb.init(project="GRPO-V1", sync_tensorboard=True, dir="./wandb", name=name)

        self.lr_scheduler = CosineAnnealingWarmupLR(
            optimizer=self.optimizer,
            total_steps=min(self.num_episodes, 4) * self.num_update_per_episode,
            warmup_steps=0,
            eta_min=0.1 * training_config.get("lr", 1e-6),
        )

    def setup(self):
        super().setup()
        self.policy_model, self.optimizer, _, _, self.lr_scheduler = self.booster.boost(
            self.policy_model, self.optimizer, lr_scheduler=self.lr_scheduler
        )
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
        forward_batch_size = self.training_config.get("forward_micro_batch_size", data["input_ids"].size(0))

        need_update = (step_idx + 1) % self.num_microbatches == 0
        ctx = nullcontext() if need_update else self.booster.no_sync(self.policy_model, self.optimizer)
        with ctx:
            policy_model_logits = get_logits_rebatched_forward(
                self.policy_model,
                forward_batch_size,
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
            )
            action_log_probs = calc_action_log_probs(
                policy_model_logits / self.generate_config["temperature"],
                data["input_ids"],
                num_action,
                self.plugin.shard_config,
            )

            with torch.no_grad():
                reference_model_logits = get_logits_rebatched_forward(
                    self.reference_model,
                    forward_batch_size,
                    input_ids=data["input_ids"],
                    attention_mask=data["attention_mask"],
                )
            reference_action_log_probs = calc_action_log_probs(
                reference_model_logits / self.generate_config["temperature"],
                data["input_ids"],
                num_action,
                self.plugin.shard_config,
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
            reward_mean = group_reward.mean(dim=1)
            # filter out the reward that is too high (all sample gets full score) or too low (all sample gets 0 score),
            loss_mask = (
                None
                if self.filter_range is None
                else torch.logical_and(
                    reward_mean > self.filter_range[0], reward_mean < self.filter_range[1]
                ).repeat_interleave(self.num_generations, dim=0)
            )

            # [batch_size x num_generations]
            reward_mean = reward_mean.repeat_interleave(self.num_generations, dim=0)
            reward_std = group_reward.std(dim=1).repeat_interleave(self.num_generations, dim=0)
            # [batch_size x num_generations]
            advantages = (reward - reward_mean) / (reward_std + 1e-4)

            loss, skip_update, _ = self.policy_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages.unsqueeze(dim=-1).repeat_interleave(action_log_probs.size(-1), dim=-1),
                per_token_kl,
                action_mask,
                loss_mask=loss_mask,
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
                        "metrics/reward": self.accum_reward.item() / self.accum_count,
                        "metrics/format_reward": self.accum_format_reward.item() / self.accum_count,
                        "metrics/acc_reward": self.accum_acc_reward.item() / self.accum_count,
                        "metrics/response_length": self.accum_response_length.item() / self.accum_count,
                        "train/loss": self.accum_loss.item() / self.accum_count,
                        "train/kl": self.accum_kl.item() / self.accum_count,
                        "train/advantages": self.accum_advantages.item() / self.accum_count,
                        "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "rollout/temperature": data["temperature"].cpu().numpy()[0][0],
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


@ray.remote
class GRPOEvalConsumer(BaseConsumer):
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
        log_dir="./results",
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
        self.accum_reward = torch.zeros(1, device=self.device)
        self.accum_format_reward = torch.zeros(1, device=self.device)
        self.accum_acc_reward = torch.zeros(1, device=self.device)
        self.accum_response_length = torch.zeros(1, device=self.device)
        self.accum_count = torch.zeros(1, device=self.device)

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

        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        else:
            os.system(f"rm -rf {self.log_dir}/*")

    def setup(self):
        super().setup()
        self.policy_model, _, *_ = self.booster.boost(self.policy_model)

    def step(self, step_idx: int, **kwargs) -> Optional[float]:
        rank = dist.get_rank()
        data = {k: v.view(-1, v.size(-1)).cpu() for k, v in kwargs.items()}
        kwargs["input_ids"].size(0)
        reward_group = self.reward_model(
            data["input_ids"], gt_answer=data["gt_answer"], response_idx=data["response_idx"]
        )
        reward = [value[0].item() for value in reward_group]
        format_reward = [value[1].item() for value in reward_group]
        acc_reward = [value[2].item() for value in reward_group]
        response_length = [(data["response_idx"][i][1] - data["response_idx"][i][0]).item() for i in range(len(reward))]

        response = self.tokenizer.batch_decode(data["input_ids"], skip_special_tokens=True)
        with open(f"{self.log_dir}/eval_results_rank_{rank}.jsonl", "a", encoding="utf8") as f:
            for i in range(len(response)):
                f.write(
                    json.dumps(
                        {
                            "response": response[i],
                            "reward": reward[i],
                            "format_reward": format_reward[i],
                            "acc_reward": acc_reward[i],
                            "response_length": response_length[i],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        self.accum_reward += sum(reward)
        self.accum_format_reward += sum(format_reward)
        self.accum_acc_reward += sum(acc_reward)
        self.accum_response_length += sum(response_length)
        self.accum_count += len(reward)

        # print results
        total_count = all_reduce_mean(self.accum_count, self.plugin)
        mean_reward = all_reduce_mean(self.accum_reward, self.plugin) / total_count
        mean_format_reward = all_reduce_mean(self.accum_format_reward, self.plugin) / total_count
        mean_acc_reward = all_reduce_mean(self.accum_acc_reward, self.plugin) / total_count
        mean_response_length = all_reduce_mean(self.accum_response_length, self.plugin) / total_count
        if rank == 0:
            print(
                f"Step {step_idx}: Mean Reward: {mean_reward}, Mean Format Reward: {mean_format_reward}, Mean Acc Reward: {mean_acc_reward}, Mean Response Length: {mean_response_length}"
            )
        return None

    def state_dict(self):
        self.policy_model._force_wait_all_gather()
        model = self.policy_model.unwrap()
        state_dict = model.state_dict()
        return state_dict
