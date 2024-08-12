"""
Reward model trianer
"""

import os
from typing import Any, Callable, Optional

import torch
import tqdm
from coati.models import LogSigLoss
from coati.trainer.utils import all_reduce_mean
from coati.utils import AccumulativeMeanMeter, save_checkpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from colossalai.booster import Booster, Plugin
from colossalai.cluster import DistCoordinator
from colossalai.utils import get_current_device

from .base import SLTrainer
from .utils import is_rank_0, to_device


class RewardModelTrainer(SLTrainer):
    """
        Trainer for PPO algorithm.

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
        model: Any,
        booster: Booster,
        optimizer: Optimizer,
        plugin: Plugin,
        lr_scheduler: _LRScheduler,
        tokenizer: PreTrainedTokenizerBase,
        loss_fn: Optional[Callable] = None,
        max_epochs: int = 1,
        beta: float = 0.1,
        accumulation_steps: int = 1,
        start_epoch: int = 0,
        save_interval: int = 0,
        save_dir: str = None,
        coordinator: DistCoordinator = None,
    ) -> None:
        super().__init__(
            booster, max_epochs=max_epochs, model=model, optimizer=optimizer, plugin=plugin, start_epoch=start_epoch
        )
        self.actor_scheduler = lr_scheduler
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn if loss_fn is not None else LogSigLoss(beta=beta)
        self.save_interval = save_interval
        self.coordinator = coordinator
        self.save_dir = save_dir
        self.num_train_step = 0
        self.accumulation_steps = accumulation_steps
        self.device = get_current_device()
        self.accumulative_meter = AccumulativeMeanMeter()

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

            self.wandb_run = wandb.init(project="Coati-rm", sync_tensorboard=True)
        if log_dir is not None and is_rank_0():
            import os
            import time

            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(log_dir, "rm")
            log_dir = os.path.join(log_dir, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
            self.writer = SummaryWriter(log_dir=log_dir)

    def _train(self, epoch):
        self.model.train()
        step_bar = tqdm.trange(
            len(self.train_dataloader) // self.accumulation_steps,
            desc=f"Epoch {epoch + 1}/{self.max_epochs}",
            disable=not is_rank_0(),
        )
        for i, batch in enumerate(self.train_dataloader):
            batch = to_device(batch, self.device)

            (
                chosen_input_ids,
                chosen_attention_mask,
                reject_input_ids,
                reject_attention_mask,
            ) = (
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["reject_input_ids"],
                batch["reject_attention_mask"],
            )
            batch_size = chosen_input_ids.size()[0]

            # Concatenate for better parrallelism
            reward = self.model(
                torch.cat([chosen_input_ids, reject_input_ids], dim=0),
                attention_mask=torch.cat([chosen_attention_mask, reject_attention_mask], dim=0),
            )
            chosen_reward = reward[:batch_size]
            reject_reward = reward[batch_size:]
            loss = self.loss_fn(chosen_reward, reject_reward).mean()

            self.booster.backward(loss=loss, optimizer=self.optimizer)

            accuracy = (chosen_reward > reject_reward).float()

            # Sync
            loss_mean = all_reduce_mean(tensor=loss)
            chosen_rewards_mean = all_reduce_mean(tensor=chosen_reward)
            rejected_rewards_mean = all_reduce_mean(tensor=reject_reward)
            accuracy_mean = all_reduce_mean(tensor=accuracy)
            self.accumulative_meter.add("chosen_rewards", chosen_rewards_mean.to(torch.float16).mean().item())
            self.accumulative_meter.add("rejected_rewards", rejected_rewards_mean.to(torch.float16).mean().item())
            self.accumulative_meter.add("loss", loss_mean.to(torch.float16).item())
            self.accumulative_meter.add("accuracy", accuracy_mean.mean().to(torch.float16).item())

            if (i + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.actor_scheduler.step()
                step_bar.update()
                self.num_train_step += 1

                # Logging
                if self.writer and is_rank_0():
                    self.writer.add_scalar("train/loss", self.accumulative_meter.get("loss"), self.num_train_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.num_train_step)
                    self.writer.add_scalar(
                        "train/dist",
                        self.accumulative_meter.get("chosen_rewards") - self.accumulative_meter.get("rejected_rewards"),
                        self.num_train_step,
                    )
                    self.writer.add_scalar(
                        "train/reward_chosen", self.accumulative_meter.get("chosen_rewards"), self.num_train_step
                    )
                    self.writer.add_scalar(
                        "train/reward_reject", self.accumulative_meter.get("rejected_rewards"), self.num_train_step
                    )
                    self.writer.add_scalar("train/acc", self.accumulative_meter.get("accuracy"), self.num_train_step)

                self.accumulative_meter.reset()

                # Save checkpoint
                if self.save_interval > 0 and (self.num_train_step + 1) % self.save_interval == 0:
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
                        f"Saved checkpoint at epoch {epoch} step {(i + 1)/self.accumulation_steps} at folder {self.save_dir}"
                    )
        step_bar.close()

    def _eval(self, epoch):
        if self.eval_dataloader is None:
            self.coordinator.print_on_master("No eval dataloader is provided, skip evaluation")
            return
        self.model.eval()
        step_bar = tqdm.trange(
            len(self.eval_dataloader), desc=f"Epoch {epoch + 1}/{self.max_epochs}", disable=not is_rank_0()
        )
        with torch.no_grad():
            for i, batch in enumerate(self.eval_dataloader):
                batch = to_device(batch, self.device)
                (
                    chosen_input_ids,
                    chosen_attention_mask,
                    reject_input_ids,
                    reject_attention_mask,
                ) = (
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["reject_input_ids"],
                    batch["reject_attention_mask"],
                )

                chosen_reward = self.model(chosen_input_ids, attention_mask=chosen_attention_mask)
                reject_reward = self.model(reject_input_ids, attention_mask=reject_attention_mask)
                loss = self.loss_fn(chosen_reward, reject_reward).mean()

                # Sync
                loss_mean = all_reduce_mean(tensor=loss)
                chosen_rewards_mean = all_reduce_mean(tensor=chosen_reward)
                rejected_rewards_mean = all_reduce_mean(tensor=reject_reward)
                self.accumulative_meter.add("chosen_rewards", chosen_rewards_mean.to(torch.float16).mean().item())
                self.accumulative_meter.add("rejected_rewards", rejected_rewards_mean.to(torch.float16).mean().item())
                self.accumulative_meter.add("loss", loss_mean.to(torch.float16).item())

                step_bar.update()

            msg = "Evaluation Result:\n"
            for tag in ["loss", "chosen_rewards", "rejected_rewards"]:
                msg = msg + f"{tag}: {self.accumulative_meter.get(tag)}\n"
            msg = (
                msg
                + f"distance: {self.accumulative_meter.get('chosen_rewards')-self.accumulative_meter.get('rejected_rewards')}\n"
            )
            self.coordinator.print_on_master(msg)
            os.makedirs(self.save_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, f"eval_result_epoch{epoch}.txt"), "w") as f:
                f.write(msg)
            step_bar.close()
