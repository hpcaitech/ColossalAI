"""
Dpo trainer
"""

import os
from typing import Any, Optional

import torch
from coati.models.loss import DpoLoss
from coati.models.utils import calc_masked_log_probs
from coati.trainer.utils import all_reduce_mean
from coati.utils import AccumulativeMeanMeter, save_checkpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import PreTrainedTokenizerBase

from colossalai.booster import Booster, Plugin
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
        if use_wandb and is_rank_0():
            assert log_dir is not None, "log_dir must be provided when use_wandb is True"
            import wandb

            self.wandb_run = wandb.init(project="Coati-dpo", sync_tensorboard=True)
        if log_dir is not None and is_rank_0():
            import os
            import time

            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(log_dir, "dpo")
            log_dir = os.path.join(log_dir, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
            self.writer = SummaryWriter(log_dir=log_dir)

    def _train(self, epoch: int):
        """
        Args:
            epoch int: the number of current epoch
        """
        self.model.train()
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

            losses, chosen_rewards, rejected_rewards = self.actor_loss_fn(
                logprob_actor_chosen,
                logprob_actor_reject,
                logprob_ref_chosen if logprob_ref_chosen is not None else None,
                logprob_ref_reject if logprob_ref_reject is not None else None,
                chosen_loss_mask[:, 1:],
                reject_loss_mask[:, 1:],
            )
            reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()

            # DPO Loss
            loss = losses.mean()

            self.booster.backward(loss=loss, optimizer=self.optimizer)
            if self.num_train_step % self.accumulation_steps == self.accumulation_steps - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.actor_scheduler.step()

            # sync
            loss_mean = all_reduce_mean(tensor=loss)
            chosen_rewards_mean = all_reduce_mean(tensor=chosen_rewards)
            rejected_rewards_mean = all_reduce_mean(tensor=rejected_rewards)
            reward_accuracies_mean = all_reduce_mean(tensor=reward_accuracies)
            self.accumulative_meter.add("chosen_rewards", chosen_rewards_mean.to(torch.float16).mean().item())
            self.accumulative_meter.add("rejected_rewards", rejected_rewards_mean.to(torch.float16).mean().item())
            self.accumulative_meter.add("loss", loss_mean.to(torch.float16).item())
            self.accumulative_meter.add("accuracy", reward_accuracies_mean.to(torch.float16).item())

            if i % self.accumulation_steps == self.accumulation_steps - 1:
                self.num_train_step += 1
                step_bar.update()
                # logging
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
                        self.accumulative_meter.get("chosen_rewards") - self.accumulative_meter.get("rejected_rewards"),
                        self.num_train_step,
                    )
                    self.writer.add_scalar(
                        "train/accuracy",
                        self.accumulative_meter.get("accuracy"),
                        self.num_train_step,
                    )
                self.accumulative_meter.reset()

                if self.save_dir is not None and (self.num_train_step + 1) % self.save_interval == 0:
                    # save checkpoint
                    self.coordinator.print_on_master("\nStart saving model checkpoint with running states")
                    save_checkpoint(
                        save_dir=self.save_dir,
                        booster=self.booster,
                        model=self.model,
                        optimizer=self.optimizer,
                        lr_scheduler=self.actor_scheduler,
                        epoch=epoch,
                        step=i + 1,
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
        self.coordinator.print_on_master("\nStart evaluation...")

        step_bar = trange(
            len(self.eval_dataloader),
            desc=f"Epoch {epoch + 1}/{self.max_epochs}",
            disable=not is_rank_0(),
        )

        self.accumulative_meter.reset()

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

                self.ref_model.eval()

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
                self.accumulative_meter.add("rejected_rewards", rejected_rewards_mean.to(torch.float16).mean().item())
                self.accumulative_meter.add("loss", loss_mean.to(torch.float16).item())
                self.accumulative_meter.add("accuracy", reward_accuracies_mean.to(torch.float16).item())
                self.accumulative_meter.add(
                    "margin", (chosen_rewards_mean - rejected_rewards_mean).to(torch.float16).mean().item()
                )
                step_bar.update()

        msg = "Evaluation Result:\n"
        for tag in ["loss", "chosen_rewards", "rejected_rewards", "accuracy", "margin"]:
            msg = msg + f"{tag}: {self.accumulative_meter.get(tag)}\n"
        self.coordinator.print_on_master(msg)
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, f"eval_result_epoch{epoch}.txt"), "w") as f:
            f.write(msg)
        step_bar.close()
