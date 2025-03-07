from contextlib import nullcontext
from typing import Optional

import ray
import torch
import wandb
from coati.distributed.consumer import BaseConsumer
from coati.distributed.loss import PolicyLoss, ValueLoss
from coati.distributed.reward.reward_fn import math_reward_fn
from coati.distributed.reward.verifiable_reward import VerifiableReward
from coati.distributed.utils import calc_action_log_probs, compute_reward_ppo
from coati.trainer.utils import all_reduce_mean
from coati.models import Critic, disable_dropout
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
        num_generations=1,
        gamma:float=1.0,
        lam:float=0.95,
        kl_coef:float=0.05,
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
        self.gamma = gamma
        self.lam = lam
        self.kl_coef = kl_coef
        
        path = model_config.pop("path")
        self.policy_model = AutoModelForCausalLM.from_pretrained(path, **model_config)
        self.policy_model.train()
        self.policy_model.gradient_checkpointing_enable()
        self.critic_model = Critic(path, **model_config)
        self.critic_model.model.gradient_checkpointing_enable()
        self.critic_model.train()
        
        # Disable dropout
        disable_dropout(self.policy_model)
        disable_dropout(self.critic_model)
        
        self.optimizer = HybridAdam(self.policy_model.parameters(), lr=1e-6)
        self.critic_optimizer = HybridAdam(self.critic_model.parameters(), lr=1e-6)
        
        self.accum_loss = torch.zeros(1, device=self.device)
        self.accum_reward = torch.zeros(1, device=self.device)
        self.accum_kl = torch.zeros(1, device=self.device)
        self.accum_advantage = torch.zeros(1, device=self.device)
        self.accum_critic_loss = torch.zeros(1, device=self.device)
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
        self.critic_loss_fn = ValueLoss()
        self.global_step = 0
        if use_wandb and self.rank == 0:
            self.wandb_run = wandb.init(project="PPO-Test", sync_tensorboard=True)

    def setup(self):
        super().setup()
        self.policy_model, self.optimizer, *_ = self.booster.boost(self.policy_model, self.optimizer)
        self.critic_model, self.critic_optimizer, *_ = self.critic_booster.boost(
            self.critic_model, self.critic_optimizer
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
        old_action_log_probs = data["action_log_probs"].detach()

        need_update = (step_idx + 1) % self.num_microbatches == 0

        ctx = nullcontext() if need_update else self.booster.no_sync(self.policy_model, self.optimizer)
        with ctx:
            policy_model_logits = self.policy_model(
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
            )["logits"]
            action_log_probs = calc_action_log_probs(policy_model_logits, data["input_ids"], num_action)

            with torch.no_grad():
                reference_model_logits = self.reference_model(
                    input_ids=data["input_ids"],
                    attention_mask=data["attention_mask"],
                )["logits"]
            reference_action_log_probs = calc_action_log_probs(reference_model_logits, data["input_ids"], num_action)

            value = self.critic_model(
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
            )
            value = value[:, -num_action -1: -1] * action_mask

            r = self.reward_model(
                data["input_ids"], gt_answer=data["gt_answer"], response_idx=data["response_idx"]
            )
            reward, kl = compute_reward_ppo(
                r, self.kl_coef, old_action_log_probs, reference_action_log_probs, action_mask=action_mask
            )
                
            # Calculate advantages
            # reference: https://github.com/huggingface/trl/blob/e3244d2d096ff1e2e248c931d06d39e165e20623/trl/trainer/ppo_trainer.py#L514C17-L523C46lastgaelam = 0
            lastgaelam = 0
            advantage_reversed = []
            for t in reversed(range(num_action)):
                nextvalues = value[:, t + 1] if t < num_action - 1 else 0.0
                delta = reward[:, t] + self.gamma * nextvalues - value[:, t]
                lastgaelam = delta + self.gamma * self.lam * lastgaelam
                advantage_reversed.append(lastgaelam)
            advantage = torch.stack(advantage_reversed[::-1], axis=1) * action_mask
            advantage = advantage.detach()
            
            # KL divergence for logging
            per_token_kl = (
                torch.exp(reference_action_log_probs - action_log_probs)
                - (reference_action_log_probs - action_log_probs)
                - 1
            )
            kl = torch.sum(per_token_kl * action_mask, dim=-1) / torch.sum(action_mask, dim=-1)

            # Calculate Loss
            loss, skip_update, _ = self.policy_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantage,
                0,  # kl is already included in the advantage
                action_mask,
            )
            
            # Critic Loss
            # Hack: use the current value to approximate the old value, should be old value mathematically
            critic_loss = self.critic_loss_fn(
                value,
                value.detach(),
                advantage,
                action_mask=action_mask,
            )

            if not skip_update:
                self.booster.backward(loss, self.optimizer)
                self.critic_booster.backward(critic_loss, self.critic_optimizer)
            loss = all_reduce_mean(loss, self.plugin)
            critic_loss = all_reduce_mean(critic_loss, self.plugin)
            r_mean = all_reduce_mean(r.mean(), self.plugin)
            kl = all_reduce_mean(kl.mean(), self.plugin)
            advantage = all_reduce_mean(advantage.mean(), self.plugin)
            self.accum_loss.add_(loss.data)
            self.accum_critic_loss.add_(critic_loss.data)
            self.accum_advantage.add_(advantage.data)
            self.accum_reward.add_(r_mean.data)
            self.accum_kl.add_(kl.data)
            self.accum_count += 1
        if self.rank == 0:
            print(f"input_ids: {data['input_ids'].shape}, reward: {r_mean.item()}")
        if need_update:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()
            loss_scalar = self.accum_loss.item()
            if self.rank == 0:
                print(
                    "Loss:",
                    self.accum_loss.item() / self.accum_count,
                    "Reward:",
                    self.accum_reward.item() / self.accum_count,
                    "KL:",
                    self.accum_kl.item() / self.accum_count,
                )
                if self.global_step % 3 == 0:
                    for i in range(min(3, data["input_ids"].shape[0])):
                        response_decoded_for_logging = self.tokenizer.decode(
                            data["input_ids"][i], skip_special_tokens=True
                        )
                        response_reward_for_logging = r[i]
                        print(f"###### Generation Sample {i} ######\nResponse: {response_decoded_for_logging}\nReward: {response_reward_for_logging}")
                self.wandb_run.log(
                    {
                        "train/loss": self.accum_loss.item() / self.accum_count,
                        "train/reward": self.accum_reward.item() / self.accum_count,
                        "train/kl": self.accum_kl.item() / self.accum_count,
                        "train/critic_loss": self.accum_critic_loss.item() / self.accum_count,
                        "train/advantage": self.accum_advantage.item() / self.accum_count,
                    }
                )
            self.accum_loss.zero_()
            self.accum_reward.zero_()
            self.accum_kl.zero_()
            self.accum_advantage.zero_()
            self.accum_critic_loss.zero_()
            self.accum_count = 0
            self.global_step += 1
            return loss_scalar

    def state_dict(self):
        self.policy_model._force_wait_all_gather()
        model = self.policy_model.unwrap()
        state_dict = model.state_dict()
        return state_dict
