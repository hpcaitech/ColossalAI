from contextlib import nullcontext
from typing import Any, Optional

import ray
import torch
import wandb
from coati.distributed.consumer import BaseConsumer
from coati.distributed.loss import PolicyLoss
from coati.distributed.utils import entropy_from_logits, memory_efficient_logprob
from coati.trainer.utils import all_reduce_mean, all_reduce_sum
from coati.utils import load_checkpoint
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
        minibatch_size=1,
        num_generations=8,
        generate_config=None,
        grpo_config={},
        save_interval: int = 100,
        save_dir="./model",
        project_name: str = None,
        run_name: str = None,
        wandb_group_name: str = None,
        enable_profiling: bool = False,
        n_behind: int = 0,
    ):
        print(f"Using GRPO config: {grpo_config}")
        if (
            plugin_config.get("pp_size", 1) > 1
            and "num_microbatches" not in plugin_config
            and "microbatch_size" not in plugin_config
        ):
            plugin_config["microbatch_size"] = max(
                1, grpo_config.get("train_microbatch_size") // plugin_config.get("pp_size", 1)
            )
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
            minibatch_size,
            save_interval=save_interval,
            save_dir=save_dir,
            enable_profiling=enable_profiling,
            n_behind=n_behind,
        )
        path = model_config.pop("path")
        self.policy_model = AutoModelForCausalLM.from_pretrained(path, **model_config)
        self.policy_model.train()
        self.policy_model.gradient_checkpointing_enable()
        self.optimizer = HybridAdam(
            self.policy_model.parameters(),
            lr=grpo_config.get("lr", 1e-6),
            weight_decay=grpo_config.get("weight_decay", 0.01),
        )
        self.accum_loss = torch.zeros(1, device=self.device)
        self.accum_kl = torch.zeros(1, device=self.device)
        self.accum_entropy = torch.zeros(1, device=self.device)
        self.accum_advantages = torch.zeros(1, device=self.device)
        self.raw_train_batch_reward = []
        self.raw_train_batch_format_acc = []
        self.raw_train_batch_ans_acc = []
        self.raw_train_batch_response_len = []
        self.accum_count = 0
        self.generate_config = generate_config
        self.grpo_config = grpo_config
        self.project_name = project_name
        self.effective_sample_count = 0
        self.effective_prompt_count = 0
        self.project_name = project_name
        self.run_name = run_name
        self.wandb_group_name = wandb_group_name

        self.policy_loss_fn = PolicyLoss(
            clip_eps_low=grpo_config.get("clip_eps_low", 0.2),
            clip_eps_high=grpo_config.get("clip_eps_high", 0.2),
            beta=grpo_config.get("beta", 0.01),
            loss_variation=grpo_config.get("loss_variation", "sample_level"),
            adv=grpo_config.get("algo"),
        )

        # Reference model is initialized from policy model.
        if self.policy_loss_fn.beta > 0:
            self.reference_model = AutoModelForCausalLM.from_pretrained(path, **model_config)
            self.reference_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.num_generations = num_generations
        self.filter_range = grpo_config.get("filter_range", None)
        if self.filter_range is not None:
            assert len(self.filter_range) == 2, "Filter range should have 2 values."

        self.filter_truncated_response = grpo_config.get("filter_truncated_response", False)
        if self.filter_truncated_response:
            self.max_length = 0
            if "max_tokens" in self.generate_config:
                self.max_length = self.generate_config["max_tokens"]
            elif "max_new_tokens" in self.generate_config:
                self.max_length = self.generate_config["max_new_tokens"]
            else:
                raise ValueError(
                    "either max_tokens (vllm) or max_new_tokens (transformers) must be set in generate_config."
                )
        # Initialize verifiable reward.
        grpo_config.get("response_format_tags", None)
        self.global_step = 0

        self.lr_scheduler = CosineAnnealingWarmupLR(
            optimizer=self.optimizer,
            total_steps=min(self.num_episodes, 4) * self.num_update_per_episode,
            warmup_steps=0,
            eta_min=0.1 * grpo_config.get("lr", 1e-6),
        )

        self.adv = grpo_config.get("algo")

    def setup(self):
        super().setup()
        if (not self.plugin.pp_size > 1 and self.rank == 0) or (
            self.plugin.pp_size > 1
            and self.booster.plugin.stage_manager.is_last_stage()
            and self.tp_rank == 0
            and self.dp_rank == 0
        ):
            self.wandb_run = wandb.init(
                project=self.project_name,
                sync_tensorboard=False,
                dir="./wandb",
                name=self.run_name,
                group=self.wandb_group_name,
            )

        self.policy_model, self.optimizer, _, _, self.lr_scheduler = self.booster.boost(
            self.policy_model, self.optimizer, lr_scheduler=self.lr_scheduler
        )
        if self.policy_loss_fn.beta > 0:
            self.reference_model, *_ = self.booster.boost(self.reference_model)
        if self.checkpoint_path is not None:
            load_checkpoint(
                self.checkpoint_path,
                self.booster,
                self.policy_model,
                self.optimizer,
                self.lr_scheduler,
            )
        self.plugin.logger.set_level("ERROR")

    def step(self, step_idx: int, pbar: Any, **kwargs) -> Optional[float]:
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
            [minibatch_size, num_of_generation, prompt_length + response_length] --- <PAD>...<PAD><PROMPT>...<PROMPT><RESPONSE>...<RESPONSE><PAD>...<PAD>.
        """
        # Reshape to [minibatch_size x num_of_generation, prompt_length + response_length]
        data = {k: v.view(-1, v.size(-1)) for k, v in kwargs.items() if "raw_train_mini_batch_" not in k}
        self.raw_train_batch_reward.extend(kwargs["raw_train_mini_batch_reward"])
        self.raw_train_batch_format_acc.extend(kwargs["raw_train_mini_batch_format_acc"])
        self.raw_train_batch_ans_acc.extend(kwargs["raw_train_mini_batch_ans_acc"])
        self.raw_train_batch_response_len.extend(kwargs["raw_train_mini_batch_response_len"])
        action_mask = data["action_mask"]
        num_action = action_mask.shape[1]
        old_action_log_probs = data["action_log_probs"]
        response_length = torch.sum(action_mask, dim=1).to(torch.float32)
        train_microbatch_size = self.grpo_config.get("train_microbatch_size", data["input_ids"].size(0))

        reward = data["reward"].view((-1))
        format_acc = data["format_acc"].view((-1))
        ans_acc = data["ans_acc"].view((-1))

        # [minibatch_size, num_generations]

        group_reward = reward.view(-1, self.num_generations)
        reward_mean = group_reward.mean(dim=1)
        # [minibatch_size x num_generations]
        reward_mean = reward_mean.repeat_interleave(self.num_generations, dim=0)

        if self.adv == "GRPO" or self.adv == "DAPO":

            reward_std = group_reward.std(dim=1).repeat_interleave(self.num_generations, dim=0)
            # [minibatch_size x num_generations]
            advantages = ((reward - reward_mean) / (reward_std + 1e-4)).unsqueeze(dim=-1)

        elif self.adv == "REINFORCE_PPB":

            # [minibatch_size x num_generations]
            advantages = ((reward - reward_mean)).unsqueeze(dim=-1)

        elif self.adv == "RLOO":

            advantages = (
                reward * self.num_generations / (self.num_generations - 1)
                - reward_mean * self.num_generations / (self.num_generations - 1)
            ).unsqueeze(dim=-1)

        # [minibatch_size x num_of_generation]
        loss_mask = torch.ones(action_mask.size(0), device=action_mask.device).bool()

        # filter out overlength samples
        if self.filter_truncated_response and action_mask.size(1) == self.max_length:
            loss_mask = torch.logical_and(
                loss_mask,
                action_mask[:, -1] == False,
            )
        if self.filter_range is not None and self.grpo_config.get("dynamic_batching", False) == False:
            # filter out samples with reward outside the range
            # if dynamic batching is enabled, we filter out out of range groups before training
            group_ans_acc_mean = (
                ans_acc.view(-1, self.num_generations).mean(dim=1).repeat_interleave(self.num_generations, dim=-1)
            )
            loss_mask = torch.logical_and(
                loss_mask,
                torch.logical_and(
                    group_ans_acc_mean > self.filter_range[0],
                    group_ans_acc_mean < self.filter_range[1],
                ),
            )
        self.effective_prompt_count += group_reward.size(0) * self.dp_size

        mean_kl, mean_loss = [], []

        if self.grpo_config.get("dynamic_batching", True):
            need_update = self.effective_prompt_count >= self.batch_size * self.dp_size
        else:
            # If dynamic batching is disabled, we need to use all samples for training.
            need_update = (step_idx + 1) % self.num_microbatches == 0

        effective_samples = all_reduce_sum(torch.sum(loss_mask), self.plugin)
        effective_tokens_count = torch.sum(action_mask, dim=-1) * loss_mask
        total_effective_tokens_count = all_reduce_sum(torch.sum(effective_tokens_count), self.plugin)
        self.effective_sample_count += effective_samples.item()
        pbar.set_postfix(
            {
                "Global Step": self.global_step,
                "Gradient Accumulation on": f"{self.effective_prompt_count}/{self.batch_size * self.dp_size} effective prompts, {self.effective_sample_count}/{self.batch_size * self.dp_size * self.num_generations} effective samples",
            }
        )

        # Gradient must be synchronized if zero2 is enabled. https://github.com/hpcaitech/ColossalAI/blob/44d4053fec005fe0b06b6bc755fdc962463145df/colossalai/booster/plugin/hybrid_parallel_plugin.py#L1500
        ctx = (
            nullcontext()
            if need_update or self.booster.plugin.zero_stage == 2
            else self.booster.no_sync(self.policy_model, self.optimizer)
        )
        with ctx:
            mini_batch_entropies = []
            for forward_micro_batch_start in range(0, data["input_ids"].size(0), train_microbatch_size):
                input_ids_forward_micro_batch = data["input_ids"][
                    forward_micro_batch_start : forward_micro_batch_start + train_microbatch_size
                ]
                old_action_log_probs_micro_batch = old_action_log_probs[
                    forward_micro_batch_start : forward_micro_batch_start + train_microbatch_size
                ]
                attention_mask_forward_micro_batch = data["attention_mask"][
                    forward_micro_batch_start : forward_micro_batch_start + train_microbatch_size
                ]
                action_mask_forward_micro_batch = action_mask[
                    forward_micro_batch_start : forward_micro_batch_start + train_microbatch_size
                ]
                loss_mask_forward_micro_batch = (
                    loss_mask[forward_micro_batch_start : forward_micro_batch_start + train_microbatch_size]
                    if loss_mask is not None
                    else None
                )
                advantages_forward_micro_batch = advantages[
                    forward_micro_batch_start : forward_micro_batch_start + train_microbatch_size
                ]

                if self.plugin.pp_size > 1:
                    # Support training with PP.
                    if self.policy_loss_fn.beta > 0:
                        with torch.no_grad():
                            reference_model_outputs = self.booster.execute_pipeline(
                                iter(
                                    [
                                        {
                                            "input_ids": input_ids_forward_micro_batch,
                                            "attention_mask": attention_mask_forward_micro_batch,
                                        }
                                    ]
                                ),
                                self.reference_model,
                                criterion=lambda outputs, inputs: torch.tensor(
                                    [0.0], device=action_mask.device
                                ),  # dummy criterion
                                optimizer=None,
                                return_loss=False,
                                return_outputs=True,
                            )

                        if self.booster.plugin.stage_manager.is_last_stage():
                            reference_action_log_probs = memory_efficient_logprob(
                                reference_model_outputs["outputs"]["logits"] / self.generate_config["temperature"],
                                input_ids_forward_micro_batch,
                                num_action,
                                shard_config=self.plugin.shard_config,
                            )
                        else:
                            # Dummy reference logprobs for data iterator.
                            reference_action_log_probs = None
                    else:
                        reference_action_log_probs = None

                    data_policy_forward = {
                        "input_ids": input_ids_forward_micro_batch,
                        "attention_mask": attention_mask_forward_micro_batch,
                        "action_mask": action_mask_forward_micro_batch,
                        "advantages": advantages_forward_micro_batch,
                        "loss_mask": loss_mask_forward_micro_batch,
                        "old_action_log_probs": old_action_log_probs_micro_batch,
                        "source": self.rank,
                    }
                    if reference_action_log_probs is not None:
                        data_policy_forward["reference_action_log_probs"] = reference_action_log_probs

                    kl = []

                    def _criterion(outputs, inputs):
                        action_logits = outputs.logits
                        mini_batch_entropies.append(
                            (
                                ((entropy_from_logits(action_logits[:, -num_action:]) * inputs["action_mask"]).sum(-1))
                                / inputs["action_mask"].sum(-1)
                            ).detach()
                        )
                        action_log_probs = memory_efficient_logprob(
                            action_logits / self.generate_config["temperature"],
                            inputs["input_ids"],
                            num_action,
                            shard_config=self.plugin.shard_config,
                        )
                        if "reference_action_log_probs" in inputs:
                            per_token_kl = (
                                torch.exp(inputs["reference_action_log_probs"] - action_log_probs)
                                - (inputs["reference_action_log_probs"] - action_log_probs)
                                - 1
                            )
                            appox_kl = torch.sum(per_token_kl * inputs["action_mask"], dim=-1) / torch.sum(
                                inputs["action_mask"], dim=-1
                            )
                            kl.append(appox_kl.mean())
                        else:
                            per_token_kl = 0.0
                            kl.append(torch.tensor(0.0))

                        inputs["advantages"].repeat_interleave(action_log_probs.size(-1), dim=-1)

                        if self.adv == "REINFORCE_PPB":

                            inputs["advantages"] = inputs["advantages"] - self.policy_loss_fn.beta * per_token_kl
                            advantages_forward_micro_batch_mean = torch.sum(
                                inputs["advantages"] * inputs["action_mask"]
                            ) / (torch.sum(inputs["action_mask"]) + 1e-4)
                            advantages_forward_micro_batch_std = torch.rsqrt(
                                torch.sum(
                                    (inputs["advantages"] - advantages_forward_micro_batch_mean) ** 2
                                    * inputs["action_mask"]
                                )
                                / (torch.sum(inputs["action_mask"]) + 1e-4)
                                + 1e-8
                            )
                            inputs["advantages"] = (
                                (inputs["advantages"] - advantages_forward_micro_batch_mean)
                                * inputs["action_mask"]
                                / (advantages_forward_micro_batch_std)
                            )

                            per_token_kl = 0.0

                        loss, _ = self.policy_loss_fn(
                            action_log_probs,
                            inputs["old_action_log_probs"],
                            inputs["advantages"],
                            per_token_kl,
                            inputs["action_mask"],
                            loss_mask=inputs["loss_mask"],
                            total_effective_tokens_in_batch=total_effective_tokens_count,
                        )
                        return loss

                    policy_model_outputs = self.booster.execute_pipeline(
                        iter([data_policy_forward]),
                        self.policy_model,
                        criterion=_criterion,
                        optimizer=self.optimizer,
                        return_loss=True,
                        return_outputs=False,
                    )
                    loss = policy_model_outputs["loss"]

                    if self.booster.plugin.stage_manager.is_last_stage():
                        if len(kl) > 0:
                            kl = all_reduce_mean(torch.mean(torch.stack(kl)).to(loss.device), self.plugin).data
                            mean_kl.append(kl)
                        mean_loss.append(all_reduce_mean(loss, self.plugin).data)
                else:
                    policy_model_logits = self.policy_model(
                        input_ids=input_ids_forward_micro_batch,
                        attention_mask=attention_mask_forward_micro_batch,
                    ).logits
                    action_log_probs = memory_efficient_logprob(
                        policy_model_logits / self.generate_config["temperature"],
                        input_ids_forward_micro_batch,
                        num_action,
                        shard_config=self.plugin.shard_config,
                    )

                    if self.policy_loss_fn.beta > 0:
                        with torch.no_grad():
                            reference_model_logits = self.reference_model(
                                input_ids=input_ids_forward_micro_batch,
                                attention_mask=attention_mask_forward_micro_batch,
                            ).logits
                        reference_action_log_probs = memory_efficient_logprob(
                            reference_model_logits / self.generate_config["temperature"],
                            input_ids_forward_micro_batch,
                            num_action,
                            shard_config=self.plugin.shard_config,
                        )
                        per_token_kl = (
                            torch.exp(reference_action_log_probs - action_log_probs)
                            - (reference_action_log_probs - action_log_probs)
                            - 1
                        )
                        kl = torch.sum(per_token_kl * action_mask_forward_micro_batch, dim=-1) / torch.sum(
                            action_mask_forward_micro_batch, dim=-1
                        )
                    else:
                        per_token_kl = 0.0
                        kl = None

                    (
                        advantages_forward_micro_batch.repeat_interleave(action_log_probs.size(-1), dim=-1)
                        - self.policy_loss_fn.beta * per_token_kl
                    )

                    if self.adv == "REINFORCE_PPB":

                        advantages_forward_micro_batch = (
                            advantages_forward_micro_batch - self.policy_loss_fn.beta * per_token_kl
                        )
                        advantages_forward_micro_batch_mean = torch.sum(
                            advantages_forward_micro_batch * action_mask_forward_micro_batch
                        ) / (torch.sum(action_mask_forward_micro_batch) + 1e-4)
                        advantages_forward_micro_batch_std = torch.rsqrt(
                            torch.sum(
                                (advantages_forward_micro_batch - advantages_forward_micro_batch_mean) ** 2
                                * action_mask_forward_micro_batch
                            )
                            / (torch.sum(action_mask_forward_micro_batch) + 1e-4)
                            + 1e-8
                        )
                        advantages_forward_micro_batch = (
                            (advantages_forward_micro_batch - advantages_forward_micro_batch_mean)
                            * action_mask_forward_micro_batch
                            / (advantages_forward_micro_batch_std)
                        )

                        per_token_kl = 0.0

                    loss, _ = self.policy_loss_fn(
                        action_log_probs,
                        old_action_log_probs_micro_batch,
                        advantages_forward_micro_batch,
                        per_token_kl,
                        action_mask_forward_micro_batch,
                        loss_mask=loss_mask_forward_micro_batch,
                        total_effective_tokens_in_batch=total_effective_tokens_count,
                    )

                    self.booster.backward(loss, self.optimizer)
                    loss = all_reduce_mean(loss, self.plugin)
                    # Calculate accumulate value.
                    if kl is not None:
                        kl = all_reduce_mean(kl.mean(), self.plugin)
                        mean_kl.append(kl.data)
                    mean_loss.append(loss.data)
                    mini_batch_entropies.append(
                        all_reduce_mean(
                            (
                                (
                                    (
                                        entropy_from_logits(policy_model_logits[:, -num_action:])
                                        * action_mask_forward_micro_batch
                                    ).sum(-1)
                                )
                                / action_mask_forward_micro_batch.sum(-1)
                            ).detach(),
                            self.plugin,
                        )
                    )
            if not self.plugin.pp_size > 1 or (
                self.plugin.pp_size > 1
                and self.booster.plugin.stage_manager.is_last_stage()
                and self.tp_rank == 0
                and self.dp_rank == 0
            ):
                reward = all_reduce_mean(reward.mean(), self.plugin)
                format_acc = all_reduce_mean(format_acc.mean(), self.plugin)
                ans_acc = all_reduce_mean(ans_acc.mean(), self.plugin)
                advantages = all_reduce_mean(advantages.mean(), self.plugin)
                response_length = all_reduce_mean(response_length.mean(), self.plugin)
                entropy = all_reduce_mean(torch.cat(mini_batch_entropies, dim=0).mean(), self.plugin)
                self.accum_loss.add_(sum(mean_loss) / len(mean_loss))
                self.accum_entropy.add_(entropy.data)
                if self.policy_loss_fn.beta > 0:
                    self.accum_kl.add_(sum(mean_kl) / len(mean_kl))
                self.accum_advantages.add_(advantages.data)
                self.accum_count += 1
        if need_update:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            # no need to run all reduce as raw_train_batch_* are not splited across dp rank
            sample_utilization = self.effective_sample_count / len(self.raw_train_batch_reward) / self.num_generations
            self.effective_prompt_count = 0
            self.effective_sample_count = 0
            loss_scalar = self.accum_loss.item()
            if not self.plugin.pp_size > 1 or (
                self.plugin.pp_size > 1 and self.booster.plugin.stage_manager.is_last_stage() and self.tp_rank == 0
            ):
                if (not self.plugin.pp_size > 1 and self.rank == 0) or (
                    self.plugin.pp_size > 1 and self.booster.plugin.stage_manager.is_last_stage() and self.tp_rank == 0
                ):
                    raw_batch_reward_mean = torch.cat(self.raw_train_batch_reward, dim=0).mean().cpu().item()
                    raw_batch_format_acc_mean = torch.cat(self.raw_train_batch_format_acc, dim=0).mean().cpu().item()
                    raw_batch_ans_acc_mean = torch.cat(self.raw_train_batch_ans_acc, dim=0).mean().cpu().item()
                    raw_batch_response_len = torch.cat(self.raw_train_batch_response_len, dim=0)
                    raw_batch_response_len_mean = raw_batch_response_len.mean().cpu().item()
                    overlength_samples_ratio = (
                        (raw_batch_response_len >= action_mask.size(-1)).to(float).mean().cpu().item()
                    )  # not an exact figure, but a close estimate
                    self.raw_train_batch_reward = []
                    self.raw_train_batch_format_acc = []
                    self.raw_train_batch_ans_acc = []
                    self.raw_train_batch_response_len = []
                    to_log_msg = [
                        f"Loss: {self.accum_loss.item() / self.accum_count:.4f}",
                        f"Reward: {raw_batch_reward_mean:.4f}",
                        f"format Reward: {raw_batch_format_acc_mean:.4f}",
                        f"Acc Reward: {raw_batch_ans_acc_mean:.4f}",
                        f"Advantages: {self.accum_advantages.item() / self.accum_count:.4f}",
                        f"Response Length: {raw_batch_response_len_mean:.4f}",
                        f"Sample_utilization: {sample_utilization:.4f}",
                        f"Overlength samples ratio: {overlength_samples_ratio:.4f}",
                        f"Entropy: {self.accum_entropy.item() / self.accum_count:.4f}",
                    ] + ([f"KL: {self.accum_kl.item() / self.accum_count:.4f}"] if self.policy_loss_fn.beta > 0 else [])
                    print("\n".join(to_log_msg))
                    metrics = {
                        "metrics/reward": raw_batch_reward_mean,
                        "metrics/format_acc": raw_batch_format_acc_mean,
                        "metrics/ans_acc": raw_batch_ans_acc_mean,
                        "metrics/response_length": raw_batch_response_len_mean,
                        "train/loss": self.accum_loss.item() / self.accum_count,
                        "train/advantages": self.accum_advantages.item() / self.accum_count,
                        "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "train/sample_utilization": sample_utilization,
                        "train/entropy": self.accum_entropy.item() / self.accum_count,
                        "train/overlength_samples_ratio": overlength_samples_ratio,
                        "rollout/temperature": data["temperature"].cpu().numpy()[0][0],
                    }
                    if self.policy_loss_fn.beta > 0:
                        metrics["train/kl"] = self.accum_kl.item() / self.accum_count
                    if self.wandb_run is not None:
                        self.wandb_run.log(metrics)
                self.accum_loss.zero_()
                self.accum_kl.zero_()
                self.accum_entropy.zero_()
                self.accum_advantages.zero_()
                self.accum_count = 0
            return loss_scalar
        else:
            return None

    def state_dict(self):
        self.policy_model._force_wait_all_gather()
        model = self.policy_model.unwrap()
        state_dict = model.state_dict()
        state_dict["consumer_global_step"] = torch.tensor([self.global_step], device=self.device)
        return state_dict
