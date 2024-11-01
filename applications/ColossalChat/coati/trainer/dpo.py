"""
Dpo trainer
"""

import os
from typing import Any, Optional

import torch
import torch.distributed as dist
from coati.models.loss import DpoLoss
from coati.models.utils import calc_masked_log_probs
from coati.trainer.utils import all_reduce_mean
from coati.utils import AccumulativeMeanMeter, save_checkpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import PreTrainedTokenizerBase

from colossalai.booster import Booster, Plugin
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.utils import get_current_device

from .base import SLTrainer
from .utils import is_rank_0, to_device


class DPOTrainer(SLTrainer):
    """
        Trainer for DPO algorithm.

    Args:
        actor (Actor): the actor model in ppo algorithm
        ref_model (Critic): the reference model in ppo algorithm
        booster (Strategy): the strategy to use for training
        actor_optim (Optimizer): the optimizer to use for actor model
        actor_lr_scheduler (_LRScheduler): the lr scheduler to use for actor model
        tokenizer (PreTrainedTokenizerBase): the tokenizer to use for encoding
        max_epochs (int, defaults to 1): the max number of epochs to train
        beta (float, defaults to 0.1): the beta parameter in dpo loss
        accumulation_steps (int): the number of steps to accumulate gradients
        start_epoch (int, defaults to 0): the start epoch, non-zero if resumed from a checkpoint
        save_interval (int): the interval to save model checkpoints, default to 0, which means no checkpoint will be saved during trainning
        save_dir (str): the directory to save checkpoints
        coordinator (DistCoordinator): the coordinator to use for distributed logging
    """

    def __init__(
        self,
        actor: Any,
        ref_model: Any,
        booster: Booster,
        actor_optim: Optimizer,
        plugin: Plugin,
        actor_lr_scheduler: _LRScheduler,
        tokenizer: PreTrainedTokenizerBase,
        max_epochs: int = 1,
        beta: float = 0.1,
        gamma: float = 0.0,
        length_normalization: bool = False,
        apply_loss_mask: bool = True,
        accumulation_steps: int = 1,
        start_epoch: int = 0,
        save_interval: int = 0,
        save_dir: str = None,
        coordinator: DistCoordinator = None,
    ) -> None:
        super().__init__(
            booster, max_epochs=max_epochs, model=actor, optimizer=actor_optim, plugin=plugin, start_epoch=start_epoch
        )
        self.ref_model = ref_model
        self.actor_scheduler = actor_lr_scheduler
        self.tokenizer = tokenizer
        self.actor_loss_fn = DpoLoss(beta, gamma)
        self.apply_loss_mask = apply_loss_mask
        self.save_interval = save_interval
        self.coordinator = coordinator
        self.save_dir = save_dir
        self.num_train_step = 0
        self.accumulation_steps = accumulation_steps
        self.device = get_current_device()
        self.accumulative_meter = AccumulativeMeanMeter()
        self.length_normalization = length_normalization

    def _before_fit(
        self,
        train_preference_dataloader: DataLoader = None,
        eval_preference_dataloader: DataLoader = None,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
    ):
        """
        Args:
            prompt_dataloader (DataLoader): the dataloader to use for prompt data
            pretrain_dataloader (DataLoader): the dataloader to use for pretrain data
        """
        self.train_dataloader = train_preference_dataloader
        self.eval_dataloader = eval_preference_dataloader
        self.writer = None

        init_criterion = (
            dist.get_rank() == dist.get_world_size() - 1
            if isinstance(self.plugin, HybridParallelPlugin) and self.plugin.pp_size > 1
            else is_rank_0()
        )

        if use_wandb and init_criterion:
            assert log_dir is not None, "log_dir must be provided when use_wandb is True"
            import wandb

            self.wandb_run = wandb.init(project="Coati-dpo", sync_tensorboard=True)
        if log_dir is not None and init_criterion:
            import os
            import time

            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(log_dir, "DPO")
            log_dir = os.path.join(log_dir, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
            self.writer = SummaryWriter(log_dir=log_dir)

    def _train(self, epoch: int):
        """
        Args:
            epoch int: the number of current epoch
        """
        self.model.train()
        if isinstance(self.plugin, HybridParallelPlugin) and self.plugin.pp_size > 1:
            step_bar = tqdm(
                range(len(self.train_dataloader)),
                desc="Step",
                disable=not (dist.get_rank() == dist.get_world_size() - 1),
            )
            for i, batch in enumerate(self.train_dataloader):
                batch = to_device(batch, self.device)
                (
                    chosen_input_ids,
                    chosen_attention_mask,
                    chosen_loss_mask,
                    reject_input_ids,
                    reject_attention_mask,
                    reject_loss_mask,
                ) = (
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_loss_mask"],
                    batch["reject_input_ids"],
                    batch["reject_attention_mask"],
                    batch["reject_loss_mask"],
                )
                batch_size = chosen_input_ids.size()[0]
                # Calculate logits from reference model.
                if self.ref_model is not None:
                    self.ref_model.eval()
                    with torch.no_grad():
                        ref_all_logits = self.ref_model(
                            input_ids=torch.cat([chosen_input_ids, reject_input_ids]),
                            attention_mask=torch.cat([chosen_attention_mask, reject_attention_mask]),
                        )["logits"]
                        ref_chosen_logits = ref_all_logits[:batch_size]
                        ref_reject_logits = ref_all_logits[batch_size:]
                        logprob_ref_chosen = calc_masked_log_probs(
                            ref_chosen_logits, chosen_input_ids, chosen_loss_mask[:, 1:], self.length_normalization
                        )
                        logprob_ref_reject = calc_masked_log_probs(
                            ref_reject_logits, reject_input_ids, reject_loss_mask[:, 1:], self.length_normalization
                        )
                else:
                    logprob_ref_chosen = None
                    logprob_ref_reject = None

                # Merge chosen and reject
                inputs_ids = torch.stack([item for tup in zip(chosen_input_ids, reject_input_ids) for item in tup])
                attention_mask = torch.stack(
                    [item for tup in zip(chosen_attention_mask, reject_attention_mask) for item in tup]
                )
                loss_mask = torch.stack([item for tup in zip(chosen_loss_mask, reject_loss_mask) for item in tup])
                logprob_ref = torch.stack([item for tup in zip(logprob_ref_chosen, logprob_ref_reject) for item in tup])

                data_iter = iter(
                    [
                        {
                            "input_ids": inputs_ids,
                            "attention_mask": attention_mask,
                            "loss_mask": loss_mask,
                            "logprob_ref": logprob_ref,
                        }
                    ]
                )
                rewards = []

                def _criterion(outputs, inputs):
                    loss, chosen_rewards, rejected_rewards = self.actor_loss_fn(
                        calc_masked_log_probs(
                            outputs["logits"][0::2],
                            inputs["input_ids"][0::2],
                            inputs["loss_mask"][0::2][:, 1:],
                            self.length_normalization,
                        ),
                        calc_masked_log_probs(
                            outputs["logits"][1::2],
                            inputs["input_ids"][1::2],
                            inputs["loss_mask"][1::2][:, 1:],
                            self.length_normalization,
                        ),
                        inputs["logprob_ref"][0::2] if inputs["logprob_ref"] is not None else None,
                        inputs["logprob_ref"][1::2] if inputs["logprob_ref"] is not None else None,
                        inputs["loss_mask"][0::2][:, 1:],
                        inputs["loss_mask"][1::2][:, 1:],
                    )
                    rewards.append(chosen_rewards)
                    rewards.append(rejected_rewards)
                    return loss

                outputs = self.booster.execute_pipeline(
                    data_iter,
                    self.model,
                    criterion=_criterion,
                    optimizer=self.optimizer,
                    return_loss=True,
                )
                loss = outputs["loss"]
                if self.booster.plugin.stage_manager.is_last_stage():
                    chosen_rewards, rejected_rewards = rewards[0], rewards[1]
                    global_loss = all_reduce_mean(loss, self.plugin)
                    if dist.get_rank() == dist.get_world_size() - 1:
                        step_bar.set_postfix(
                            {
                                "train/loss": global_loss.item(),
                                "train/lr": self.actor_scheduler.get_last_lr()[0],
                                "train/chosen_rewards": chosen_rewards.to(torch.float16).mean().item(),
                                "train/rejected_rewards": rejected_rewards.to(torch.float16).mean().item(),
                            }
                        )
                        step_bar.update()
                        self.accumulative_meter.add("loss", global_loss.item())
                        self.accumulative_meter.add("chosen_rewards", chosen_rewards.to(torch.float16).mean().item())
                        self.accumulative_meter.add(
                            "rejected_rewards", rejected_rewards.to(torch.float16).mean().item()
                        )
                        if self.writer is not None:
                            self.writer.add_scalar("train/loss", self.accumulative_meter.get("loss"), i)
                            self.writer.add_scalar(
                                "train/chosen_rewards", self.accumulative_meter.get("chosen_rewards"), i
                            )
                            self.writer.add_scalar(
                                "train/rejected_rewards",
                                self.accumulative_meter.get("rejected_rewards"),
                                i,
                            )
                            self.writer.add_scalar(
                                "train/margin",
                                self.accumulative_meter.get("chosen_rewards")
                                - self.accumulative_meter.get("rejected_rewards"),
                                i,
                            )

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.actor_scheduler.step()
        else:
            self.accumulative_meter.reset()
            step_bar = trange(
                len(self.train_dataloader) // self.accumulation_steps,
                desc=f"Epoch {epoch + 1}/{self.max_epochs}",
                disable=not is_rank_0(),
            )
            for i, batch in enumerate(self.train_dataloader):
                batch = to_device(batch, self.device)
                (
                    chosen_input_ids,
                    chosen_attention_mask,
                    chosen_loss_mask,
                    reject_input_ids,
                    reject_attention_mask,
                    reject_loss_mask,
                ) = (
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_loss_mask"],
                    batch["reject_input_ids"],
                    batch["reject_attention_mask"],
                    batch["reject_loss_mask"],
                )
                if not self.apply_loss_mask:
                    chosen_loss_mask = chosen_loss_mask.fill_(1.0)
                    reject_loss_mask = reject_loss_mask.fill_(1.0)

                batch_size = chosen_input_ids.size()[0]

                actor_all_logits = self.model(
                    input_ids=torch.cat([chosen_input_ids, reject_input_ids]),
                    attention_mask=torch.cat([chosen_attention_mask, reject_attention_mask]),
                )["logits"]
                actor_chosen_logits = actor_all_logits[:batch_size]
                actor_reject_logits = actor_all_logits[batch_size:]
                logprob_actor_chosen = calc_masked_log_probs(
                    actor_chosen_logits, chosen_input_ids, chosen_loss_mask[:, 1:], self.length_normalization
                )

                logprob_actor_reject = calc_masked_log_probs(
                    actor_reject_logits, reject_input_ids, reject_loss_mask[:, 1:], self.length_normalization
                )

                if self.ref_model is not None:
                    self.ref_model.eval()
                    with torch.no_grad():
                        ref_all_logits = self.ref_model(
                            input_ids=torch.cat([chosen_input_ids, reject_input_ids]),
                            attention_mask=torch.cat([chosen_attention_mask, reject_attention_mask]),
                        )["logits"]
                        ref_chosen_logits = ref_all_logits[:batch_size]
                        ref_reject_logits = ref_all_logits[batch_size:]
                        logprob_ref_chosen = calc_masked_log_probs(
                            ref_chosen_logits, chosen_input_ids, chosen_loss_mask[:, 1:], self.length_normalization
                        )
                        logprob_ref_reject = calc_masked_log_probs(
                            ref_reject_logits, reject_input_ids, reject_loss_mask[:, 1:], self.length_normalization
                        )
                else:
                    logprob_ref_chosen = None
                    logprob_ref_reject = None

                loss, chosen_rewards, rejected_rewards = self.actor_loss_fn(
                    logprob_actor_chosen,
                    logprob_actor_reject,
                    logprob_ref_chosen if logprob_ref_chosen is not None else None,
                    logprob_ref_reject if logprob_ref_reject is not None else None,
                    chosen_loss_mask[:, 1:],
                    reject_loss_mask[:, 1:],
                )
                reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()

                self.booster.backward(loss=loss, optimizer=self.optimizer)
                # sync
                loss_mean = all_reduce_mean(tensor=loss)
                chosen_rewards_mean = all_reduce_mean(tensor=chosen_rewards)
                rejected_rewards_mean = all_reduce_mean(tensor=rejected_rewards)
                reward_accuracies_mean = all_reduce_mean(tensor=reward_accuracies)
                self.accumulative_meter.add("chosen_rewards", chosen_rewards_mean.to(torch.float16).mean().item())
                self.accumulative_meter.add("rejected_rewards", rejected_rewards_mean.to(torch.float16).mean().item())
                self.accumulative_meter.add("loss", loss_mean.to(torch.float16).item())
                self.accumulative_meter.add("accuracy", reward_accuracies_mean.to(torch.float16).item())

                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.actor_scheduler.step()

                    step_bar.set_postfix(
                        {
                            "train/loss": self.accumulative_meter.get("loss"),
                            "train/chosen_rewards": self.accumulative_meter.get("chosen_rewards"),
                            "train/rejected_rewards": self.accumulative_meter.get("rejected_rewards"),
                            "train/accuracy": self.accumulative_meter.get("accuracy"),
                        }
                    )
                    step_bar.update()
                    if self.writer and is_rank_0():
                        self.writer.add_scalar("train/loss", self.accumulative_meter.get("loss"), self.num_train_step)
                        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.num_train_step)
                        self.writer.add_scalar(
                            "train/chosen_rewards", self.accumulative_meter.get("chosen_rewards"), self.num_train_step
                        )
                        self.writer.add_scalar(
                            "train/rejected_rewards",
                            self.accumulative_meter.get("rejected_rewards"),
                            self.num_train_step,
                        )
                        self.writer.add_scalar(
                            "train/margin",
                            self.accumulative_meter.get("chosen_rewards")
                            - self.accumulative_meter.get("rejected_rewards"),
                            self.num_train_step,
                        )
                        self.writer.add_scalar(
                            "train/accuracy",
                            self.accumulative_meter.get("accuracy"),
                            self.num_train_step,
                        )
                    self.num_train_step += 1
                    self.accumulative_meter.reset()

            if self.save_dir is not None and self.num_train_step > 0 and self.num_train_step % self.save_interval == 0:
                # save checkpoint
                self.coordinator.print_on_master("\nStart saving model checkpoint with running states")
                save_checkpoint(
                    save_dir=self.save_dir,
                    booster=self.booster,
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.actor_scheduler,
                    epoch=epoch,
                    step=self.num_train_step,
                    batch_size=batch_size,
                    coordinator=self.coordinator,
                )
                self.coordinator.print_on_master(
                    f"Saved checkpoint at epoch {epoch} step {self.save_interval} at folder {self.save_dir}"
                )

        step_bar.close()

    def _eval(self, epoch: int):
        """
        Args:
            epoch int: the number of current epoch
        """
        if self.eval_dataloader is None:
            self.coordinator.print_on_master("No eval dataloader is provided, skip evaluation")
            return
        self.model.eval()
        self.ref_model.eval()
        self.accumulative_meter.reset()
        self.coordinator.print_on_master("\nStart evaluation...")

        if isinstance(self.plugin, HybridParallelPlugin) and self.plugin.pp_size > 1:
            step_bar = tqdm(
                range(len(self.eval_dataloader)),
                desc="Step",
                disable=not (dist.get_rank() == dist.get_world_size() - 1),
            )
            with torch.no_grad():
                for _, batch in enumerate(self.eval_dataloader):
                    batch = to_device(batch, self.device)
                    (
                        chosen_input_ids,
                        chosen_attention_mask,
                        chosen_loss_mask,
                        reject_input_ids,
                        reject_attention_mask,
                        reject_loss_mask,
                    ) = (
                        batch["chosen_input_ids"],
                        batch["chosen_attention_mask"],
                        batch["chosen_loss_mask"],
                        batch["reject_input_ids"],
                        batch["reject_attention_mask"],
                        batch["reject_loss_mask"],
                    )
                    batch_size = chosen_input_ids.size()[0]
                    # Calculate logits from reference model.
                    if self.ref_model is not None:
                        self.ref_model.eval()
                        with torch.no_grad():
                            ref_all_logits = self.ref_model(
                                input_ids=torch.cat([chosen_input_ids, reject_input_ids]),
                                attention_mask=torch.cat([chosen_attention_mask, reject_attention_mask]),
                            )["logits"]
                            ref_chosen_logits = ref_all_logits[:batch_size]
                            ref_reject_logits = ref_all_logits[batch_size:]
                            logprob_ref_chosen = calc_masked_log_probs(
                                ref_chosen_logits, chosen_input_ids, chosen_loss_mask[:, 1:], self.length_normalization
                            )
                            logprob_ref_reject = calc_masked_log_probs(
                                ref_reject_logits, reject_input_ids, reject_loss_mask[:, 1:], self.length_normalization
                            )
                    else:
                        logprob_ref_chosen = None
                        logprob_ref_reject = None

                    # Merge chosen and reject
                    inputs_ids = torch.stack([item for tup in zip(chosen_input_ids, reject_input_ids) for item in tup])
                    attention_mask = torch.stack(
                        [item for tup in zip(chosen_attention_mask, reject_attention_mask) for item in tup]
                    )
                    loss_mask = torch.stack([item for tup in zip(chosen_loss_mask, reject_loss_mask) for item in tup])
                    logprob_ref = torch.stack(
                        [item for tup in zip(logprob_ref_chosen, logprob_ref_reject) for item in tup]
                    )

                    data_iter = iter(
                        [
                            {
                                "input_ids": inputs_ids,
                                "attention_mask": attention_mask,
                                "loss_mask": loss_mask,
                                "logprob_ref": logprob_ref,
                            }
                        ]
                    )
                    rewards = []

                    def _criterion(outputs, inputs):
                        loss, chosen_rewards, rejected_rewards = self.actor_loss_fn(
                            calc_masked_log_probs(
                                outputs["logits"][0::2],
                                inputs["input_ids"][0::2],
                                inputs["loss_mask"][0::2][:, 1:],
                                self.length_normalization,
                            ),
                            calc_masked_log_probs(
                                outputs["logits"][1::2],
                                inputs["input_ids"][1::2],
                                inputs["loss_mask"][1::2][:, 1:],
                                self.length_normalization,
                            ),
                            inputs["logprob_ref"][0::2] if inputs["logprob_ref"] is not None else None,
                            inputs["logprob_ref"][1::2] if inputs["logprob_ref"] is not None else None,
                            inputs["loss_mask"][0::2][:, 1:],
                            inputs["loss_mask"][1::2][:, 1:],
                        )
                        rewards.append(chosen_rewards)
                        rewards.append(rejected_rewards)
                        return loss

                    outputs = self.booster.execute_pipeline(
                        data_iter,
                        self.model,
                        criterion=_criterion,
                        optimizer=self.optimizer,
                        return_loss=True,
                    )
                    loss = outputs["loss"]
                    if self.booster.plugin.stage_manager.is_last_stage():
                        chosen_rewards, rejected_rewards = rewards[0], rewards[1]
                        global_loss = all_reduce_mean(loss, self.plugin)
                        chosen_rewards_mean = all_reduce_mean(chosen_rewards, self.plugin)
                        rejected_rewards_mean = all_reduce_mean(rejected_rewards, self.plugin)
                        if dist.get_rank() == dist.get_world_size() - 1:
                            step_bar.set_postfix(
                                {
                                    "eval/loss": global_loss.item(),
                                    "eval/lr": self.actor_scheduler.get_last_lr()[0],
                                    "eval/chosen_rewards": chosen_rewards.to(torch.float16).mean().item(),
                                    "eval/rejected_rewards": rejected_rewards.to(torch.float16).mean().item(),
                                }
                            )
                            self.accumulative_meter.add(
                                "chosen_rewards", chosen_rewards_mean.to(torch.float16).mean().item()
                            )
                            self.accumulative_meter.add(
                                "rejected_rewards", rejected_rewards_mean.to(torch.float16).mean().item()
                            )
                            self.accumulative_meter.add("loss", global_loss.to(torch.float16).item())
                            step_bar.update()
                if self.booster.plugin.stage_manager.is_last_stage():
                    msg = "\nEvaluation Result:\n"
                    for tag in ["loss", "chosen_rewards", "rejected_rewards"]:
                        msg = msg + f"{tag}: {self.accumulative_meter.get(tag)}\n"
                    if dist.get_rank() == dist.get_world_size() - 1:
                        print(msg)
        else:
            step_bar = trange(
                len(self.eval_dataloader),
                desc=f"Epoch {epoch + 1}/{self.max_epochs}",
                disable=not is_rank_0(),
            )
            with torch.no_grad():
                for i, batch in enumerate(self.eval_dataloader):
                    batch = to_device(batch, self.device)
                    (
                        chosen_input_ids,
                        chosen_attention_mask,
                        chosen_loss_mask,
                        reject_input_ids,
                        reject_attention_mask,
                        reject_loss_mask,
                    ) = (
                        batch["chosen_input_ids"],
                        batch["chosen_attention_mask"],
                        batch["chosen_loss_mask"],
                        batch["reject_input_ids"],
                        batch["reject_attention_mask"],
                        batch["reject_loss_mask"],
                    )
                    if not self.apply_loss_mask:
                        chosen_loss_mask = chosen_loss_mask.fill_(1.0)
                        reject_loss_mask = reject_loss_mask.fill_(1.0)

                    batch_size = chosen_input_ids.size()[0]

                    actor_all_logits = self.model(
                        torch.cat([chosen_input_ids, reject_input_ids]),
                        torch.cat([chosen_attention_mask, reject_attention_mask]),
                    )["logits"]
                    actor_chosen_logits = actor_all_logits[:batch_size]
                    actor_reject_logits = actor_all_logits[batch_size:]

                    logprob_actor_chosen = calc_masked_log_probs(
                        actor_chosen_logits, chosen_input_ids, chosen_loss_mask[:, 1:], self.length_normalization
                    )

                    logprob_actor_reject = calc_masked_log_probs(
                        actor_reject_logits, reject_input_ids, reject_loss_mask[:, 1:], self.length_normalization
                    )
                    ref_all_logits = self.ref_model(
                        torch.cat([chosen_input_ids, reject_input_ids]),
                        torch.cat([chosen_attention_mask, reject_attention_mask]),
                    )["logits"]
                    ref_chosen_logits = ref_all_logits[:batch_size]
                    ref_reject_logits = ref_all_logits[batch_size:]
                    logprob_ref_chosen = calc_masked_log_probs(
                        ref_chosen_logits, chosen_input_ids, chosen_loss_mask[:, 1:], self.length_normalization
                    )
                    logprob_ref_reject = calc_masked_log_probs(
                        ref_reject_logits, reject_input_ids, reject_loss_mask[:, 1:], self.length_normalization
                    )

                    losses, chosen_rewards, rejected_rewards = self.actor_loss_fn(
                        logprob_actor_chosen,
                        logprob_actor_reject,
                        logprob_ref_chosen if logprob_ref_chosen is not None else None,
                        logprob_ref_reject if logprob_ref_reject is not None else None,
                        chosen_loss_mask[:, 1:],
                        reject_loss_mask[:, 1:],
                    )
                    reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
                    loss = losses.mean()
                    loss_mean = all_reduce_mean(tensor=loss)
                    chosen_rewards_mean = all_reduce_mean(tensor=chosen_rewards)
                    rejected_rewards_mean = all_reduce_mean(tensor=rejected_rewards)
                    reward_accuracies_mean = all_reduce_mean(tensor=reward_accuracies)
                    self.accumulative_meter.add("chosen_rewards", chosen_rewards_mean.to(torch.float16).mean().item())
                    self.accumulative_meter.add(
                        "rejected_rewards", rejected_rewards_mean.to(torch.float16).mean().item()
                    )
                    self.accumulative_meter.add("loss", loss_mean.to(torch.float16).item())
                    self.accumulative_meter.add("accuracy", reward_accuracies_mean.to(torch.float16).item())
                    self.accumulative_meter.add(
                        "margin", (chosen_rewards_mean - rejected_rewards_mean).to(torch.float16).mean().item()
                    )
                    step_bar.set_postfix(
                        {
                            "eval/loss": self.accumulative_meter.get("loss"),
                            "eval/chosen_rewards": self.accumulative_meter.get("chosen_rewards"),
                            "eval/rejected_rewards": self.accumulative_meter.get("rejected_rewards"),
                            "eval/accuracy": self.accumulative_meter.get("accuracy"),
                        }
                    )
                    step_bar.update()

            msg = "\nEvaluation Result:\n"
            for tag in ["loss", "chosen_rewards", "rejected_rewards", "accuracy", "margin"]:
                msg = msg + f"{tag}: {self.accumulative_meter.get(tag)}\n"
            self.coordinator.print_on_master(msg)
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, f"eval_result_epoch{epoch}.txt"), "w") as f:
                f.write(msg)
        step_bar.close()
