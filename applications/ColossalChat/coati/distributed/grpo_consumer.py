import json
import os
import warnings
from contextlib import nullcontext
from typing import Any, Optional

import ray
import torch
import torch.distributed as dist
import wandb
from coati.distributed.consumer import BaseConsumer
from coati.distributed.loss import PolicyLoss
from coati.distributed.reward.reward_fn import math_reward_fn
from coati.distributed.reward.verifiable_reward import VerifiableReward
from coati.distributed.utils import calc_action_log_probs
from coati.trainer.utils import all_reduce_mean, all_reduce_sum
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
        grpo_config={},
        project_name=None,
    ):
        print(f"Using GRPO config: {grpo_config}")
        if grpo_config.get("loss_variation", "sample_level") == "token_level":
            if batch_size != microbatch_size:
                warnings.warn(
                    f"Applied token_level loss, force overwrite mini-batch-size with batch-size: mini-batch-size: {microbatch_size}->{batch_size}",
                    UserWarning,
                )
                microbatch_size = batch_size
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
        self.optimizer = HybridAdam(self.policy_model.parameters(), lr=grpo_config.get("lr", 1e-6))
        self.accum_loss = torch.zeros(1, device=self.device)
        self.accum_reward = torch.zeros(1, device=self.device)
        self.accum_kl = torch.zeros(1, device=self.device)
        self.accum_format_acc = torch.zeros(1, device=self.device)
        self.accum_ans_acc = torch.zeros(1, device=self.device)
        self.accum_advantages = torch.zeros(1, device=self.device)
        self.accum_response_length = torch.zeros(1, device=self.device)
        self.accum_count = 0
        self.generate_config = generate_config
        self.grpo_config = grpo_config
        self.project_name = project_name
        self.effective_sample_count = 0
        self.total_sample_count = 0

        self.policy_loss_fn = PolicyLoss(
            clip_eps_low=grpo_config.get("clip_eps_low", 0.2),
            clip_eps_high=grpo_config.get("clip_eps_high", 0.2),
            beta=grpo_config.get("beta", 0.01),
            loss_variation=grpo_config.get("loss_variation", "sample_level"),
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
        response_format_tags = {
            "think_start": {"text": "<think>", "num_occur": 1},
            "think_end": {"text": "</think>", "num_occur": 1},
            "answer_start": {"text": "<answer>", "num_occur": 1},
            "answer_end": {"text": "</answer>", "num_occur": 1},
        }
        reward_model_kwargs = {
            k: v for k, v in grpo_config.items() if k in ["soft_over_length_punishment", "max_length", "cache_length"]
        }
        self.reward_model = VerifiableReward(
            reward_fns=[math_reward_fn], tokenizer=self.tokenizer, tags=response_format_tags, **reward_model_kwargs
        )
        self.global_step = 0
        self.use_wandb = use_wandb

        self.lr_scheduler = CosineAnnealingWarmupLR(
            optimizer=self.optimizer,
            total_steps=min(self.num_episodes, 4) * self.num_update_per_episode,
            warmup_steps=0,
            eta_min=0.1 * grpo_config.get("lr", 1e-6),
        )

    def setup(self):
        super().setup()
        if self.use_wandb and (
            (not self.plugin.pp_size > 1 and self.rank == 0)
            or (self.plugin.pp_size > 1 and self.booster.plugin.stage_manager.is_last_stage() and self.tp_rank == 0)
        ):
            # Initialize wandb.
            name = f"{self.generate_config['backend']}_bs_{self.batch_size*self.dp_size}_temp_{self.generate_config['temperature']:.01f}_top_p_{self.generate_config['top_p']:.02f}"
            self.wandb_run = wandb.init(project=self.project_name, sync_tensorboard=True, dir="./wandb", name=name)

        self.policy_model, self.optimizer, _, _, self.lr_scheduler = self.booster.boost(
            self.policy_model, self.optimizer, lr_scheduler=self.lr_scheduler
        )
        if self.policy_loss_fn.beta > 0:
            self.reference_model, *_ = self.booster.boost(self.reference_model)
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
            [batch_size, num_of_generation, prompt_length + response_length] --- <PAD>...<PAD><PROMPT>...<PROMPT><RESPONSE>...<RESPONSE><PAD>...<PAD>.
        """

        # Reshape to [batch_size x num_of_generation, prompt_length + response_length]
        data = {k: v.view(-1, v.size(-1)) for k, v in kwargs.items()}
        action_mask = data["action_mask"]
        num_action = action_mask.shape[1]
        old_action_log_probs = data["action_log_probs"]
        response_length = torch.sum(action_mask, dim=1).to(torch.float32)
        forward_batch_size = self.grpo_config.get("train_microbatch_size", data["input_ids"].size(0))

        reward_group = self.reward_model(
            data["input_ids"],
            gt_answer=data["gt_answer"],
            response_idx=data["response_idx"],
        )

        reward = torch.tensor([value[0] for value in reward_group]).to(data["input_ids"].device)
        format_acc = torch.tensor([value[1] for value in reward_group]).to(data["input_ids"].device)
        ans_acc = torch.tensor([value[2] for value in reward_group]).to(data["input_ids"].device)

        # [batch_size, num_generations]

        group_reward = reward.view(-1, self.num_generations)
        reward_mean = group_reward.mean(dim=1)
        # [batch_size x num_generations]
        reward_mean = reward_mean.repeat_interleave(self.num_generations, dim=0)

        reward_std = group_reward.std(dim=1).repeat_interleave(self.num_generations, dim=0)
        # [batch_size x num_generations]
        advantages = ((reward - reward_mean) / (reward_std + 1e-4)).unsqueeze(dim=-1)
        # filter out the reward that is too high (all sample gets full score) or too low (all sample gets 0 score),
        group_ans_acc = (
            ans_acc.view(-1, self.num_generations).mean(dim=1).repeat_interleave(self.num_generations, dim=0)
        )
        loss_mask = (
            torch.ones(action_mask.size(0), device=action_mask.device).bool()
            if self.filter_range is None
            else torch.logical_and(group_ans_acc > self.filter_range[0], group_ans_acc < self.filter_range[1])
        )
        # filter out overlength samples
        if self.filter_truncated_response and action_mask.size(1) == self.max_length:
            loss_mask = torch.logical_and(
                loss_mask,
                action_mask[:, -1] == False,
            )
        effective_tokens_count = torch.sum(action_mask, dim=-1) * loss_mask

        effective_samples = all_reduce_sum(torch.sum(loss_mask), self.plugin)
        total_effective_tokens_count = all_reduce_sum(torch.sum(effective_tokens_count), self.plugin)
        total_samples = all_reduce_sum(torch.sum(torch.ones_like(loss_mask, device=loss_mask.device)), self.plugin)
        self.effective_sample_count += effective_samples.item()
        self.total_sample_count += total_samples.item()

        mean_kl, mean_loss = [], []

        # update gradient only if at least 0.7*batch_size*num_generation valid samples are collected in case a lot of samples are invalid and got filtered out.
        # balance between efficiency and accuracy
        need_update = self.effective_sample_count >= self.batch_size * self.dp_size * self.num_generations * 0.95
        pbar.set_postfix(
            {
                "Step": self.global_step + 1,
                "Status": f"Collecting: {self.effective_sample_count}/{self.batch_size * self.dp_size * self.num_generations * 0.95}",
            }
        )

        # Gradient must be synchronized if zero2 is enabled. https://github.com/hpcaitech/ColossalAI/blob/44d4053fec005fe0b06b6bc755fdc962463145df/colossalai/booster/plugin/hybrid_parallel_plugin.py#L1500
        ctx = (
            nullcontext()
            if need_update or self.booster.plugin.zero_stage == 2
            else self.booster.no_sync(self.policy_model, self.optimizer)
        )
        with ctx:
            for forward_micro_batch_start in range(0, data["input_ids"].size(0), forward_batch_size):
                input_ids_forward_micro_batch = data["input_ids"][
                    forward_micro_batch_start : forward_micro_batch_start + forward_batch_size
                ]
                attention_mask_forward_micro_batch = data["attention_mask"][
                    forward_micro_batch_start : forward_micro_batch_start + forward_batch_size
                ]
                action_mask_forward_micro_batch = action_mask[
                    forward_micro_batch_start : forward_micro_batch_start + forward_batch_size
                ]
                loss_mask_forward_micro_batch = (
                    loss_mask[forward_micro_batch_start : forward_micro_batch_start + forward_batch_size]
                    if loss_mask is not None
                    else None
                )
                advantages_forward_micro_batch = advantages[
                    forward_micro_batch_start : forward_micro_batch_start + forward_batch_size
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
                            reference_model_logits = reference_model_outputs["outputs"]["logits"]
                            reference_action_log_probs = calc_action_log_probs(
                                reference_model_logits / self.generate_config["temperature"],
                                input_ids_forward_micro_batch,
                                num_action,
                                self.plugin.shard_config,
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
                        "source": self.rank,
                    }
                    if reference_action_log_probs is not None:
                        data_policy_forward["reference_action_log_probs"] = reference_action_log_probs

                    kl = []

                    def _criterion(outputs, inputs):
                        action_logits = outputs.logits
                        action_log_probs = calc_action_log_probs(
                            action_logits / self.generate_config["temperature"],
                            inputs["input_ids"],
                            num_action,
                            self.plugin.shard_config,
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
                            kl.append(0.0)

                        loss, _ = self.policy_loss_fn(
                            action_log_probs,
                            action_log_probs,
                            inputs["advantages"].repeat_interleave(action_log_probs.size(-1), dim=-1),
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
                        return_outputs=True,
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
                    action_log_probs = calc_action_log_probs(
                        policy_model_logits / self.generate_config["temperature"],
                        input_ids_forward_micro_batch,
                        num_action,
                        self.plugin.shard_config,
                    )

                    if self.policy_loss_fn.beta > 0:
                        with torch.no_grad():
                            reference_model_logits = self.reference_model(
                                input_ids=input_ids_forward_micro_batch,
                                attention_mask=attention_mask_forward_micro_batch,
                            ).logits
                        reference_action_log_probs = calc_action_log_probs(
                            reference_model_logits / self.generate_config["temperature"],
                            input_ids_forward_micro_batch,
                            num_action,
                            self.plugin.shard_config,
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

                    loss, _ = self.policy_loss_fn(
                        action_log_probs,
                        old_action_log_probs,
                        advantages_forward_micro_batch.repeat_interleave(action_log_probs.size(-1), dim=-1),
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
            if not self.plugin.pp_size > 1 or (
                self.plugin.pp_size > 1 and self.booster.plugin.stage_manager.is_last_stage() and self.tp_rank == 0
            ):
                reward = all_reduce_mean(reward.mean(), self.plugin)
                format_acc = all_reduce_mean(format_acc.mean(), self.plugin)
                ans_acc = all_reduce_mean(ans_acc.mean(), self.plugin)
                advantages = all_reduce_mean(advantages.mean(), self.plugin)
                response_length = all_reduce_mean(response_length.mean(), self.plugin)
                self.accum_loss.add_(sum(mean_loss) / len(mean_loss))
                if self.policy_loss_fn.beta > 0:
                    self.accum_kl.add_(sum(mean_kl) / len(mean_kl))
                self.accum_reward.add_(reward.data)
                self.accum_format_acc.add_(format_acc.data)
                self.accum_ans_acc.add_(ans_acc.data)
                self.accum_advantages.add_(advantages.data)
                self.accum_response_length.add_(response_length.data)
                self.accum_count += 1
        if need_update:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            sample_utilization = self.effective_sample_count / self.total_sample_count
            self.effective_sample_count = 0
            self.total_sample_count = 0
            if not self.plugin.pp_size > 1 or (
                self.plugin.pp_size > 1 and self.booster.plugin.stage_manager.is_last_stage() and self.tp_rank == 0
            ):
                loss_scalar = self.accum_loss.item()
                if (not self.plugin.pp_size > 1 and self.rank == 0) or (
                    self.plugin.pp_size > 1 and self.booster.plugin.stage_manager.is_last_stage() and self.tp_rank == 0
                ):
                    to_log_msg = (
                        [
                            f"Loss: {self.accum_loss.item() / self.accum_count:.4f}",
                            f"Reward: {self.accum_reward.item() / self.accum_count:.4f}",
                            f"ormat Reward: {self.accum_format_acc.item() / self.accum_count:.4f}",
                            f"Acc Reward: {self.accum_ans_acc.item() / self.accum_count:.4f}",
                            f"Advantages: {self.accum_advantages.item() / self.accum_count:.4f}",
                            f"Response Length: {self.accum_response_length.item() / self.accum_count:.4f}",
                        ]
                        + [f"KL: {self.accum_kl.item() / self.accum_count:.4f}"]
                        if self.policy_loss_fn.beta > 0
                        else []
                    )
                    print("\n".join(to_log_msg))
                    metrics = {
                        "metrics/reward": self.accum_reward.item() / self.accum_count,
                        "metrics/format_acc": self.accum_format_acc.item() / self.accum_count,
                        "metrics/ans_acc": self.accum_ans_acc.item() / self.accum_count,
                        "metrics/response_length": self.accum_response_length.item() / self.accum_count,
                        "train/loss": self.accum_loss.item() / self.accum_count,
                        "train/advantages": self.accum_advantages.item() / self.accum_count,
                        "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "train/sample_utilization": sample_utilization,
                        "rollout/temperature": data["temperature"].cpu().numpy()[0][0],
                    }
                    if self.policy_loss_fn.beta > 0:
                        metrics["train/kl"] = self.accum_kl.item() / self.accum_count

                    self.wandb_run.log(metrics)
                self.accum_loss.zero_()
                self.accum_reward.zero_()
                self.accum_ans_acc.zero_()
                self.accum_format_acc.zero_()
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
        self.accum_format_acc = torch.zeros(1, device=self.device)
        self.accum_ans_acc = torch.zeros(1, device=self.device)
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
        format_acc = [value[1].item() for value in reward_group]
        ans_acc = [value[2].item() for value in reward_group]
        response_length = [(data["response_idx"][i][1] - data["response_idx"][i][0]).item() for i in range(len(reward))]

        response = self.tokenizer.batch_decode(data["input_ids"], skip_special_tokens=True)
        with open(f"{self.log_dir}/eval_results_rank_{rank}.jsonl", "a", encoding="utf8") as f:
            for i in range(len(response)):
                f.write(
                    json.dumps(
                        {
                            "response": response[i],
                            "reward": reward[i],
                            "format_acc": format_acc[i],
                            "ans_acc": ans_acc[i],
                            "response_length": response_length[i],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        self.accum_reward += sum(reward)
        self.accum_format_acc += sum(format_acc)
        self.accum_ans_acc += sum(ans_acc)
        self.accum_response_length += sum(response_length)
        self.accum_count += len(reward)

        # print results
        total_count = all_reduce_mean(self.accum_count, self.plugin)
        mean_reward = all_reduce_mean(self.accum_reward, self.plugin) / total_count
        mean_format_acc = all_reduce_mean(self.accum_format_acc, self.plugin) / total_count
        mean_ans_acc = all_reduce_mean(self.accum_ans_acc, self.plugin) / total_count
        mean_response_length = all_reduce_mean(self.accum_response_length, self.plugin) / total_count
        if rank == 0:
            print(
                f"Step {step_idx}: Mean Reward: {mean_reward}, Mean Format Reward: {mean_format_acc}, Mean Acc Reward: {mean_ans_acc}, Mean Response Length: {mean_response_length}"
            )
        return None

    def state_dict(self):
        self.policy_model._force_wait_all_gather()
        model = self.policy_model.unwrap()
        state_dict = model.state_dict()
        return state_dict
