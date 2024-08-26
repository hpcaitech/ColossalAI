"""
KTO trainer
"""

import os
from typing import Any, Optional

import torch
import torch.distributed as dist
from coati.models.loss import KTOLoss
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


class KTOTrainer(SLTrainer):
    """
        Trainer for KTO algorithm.

    Args:
        actor (Actor): the actor model in ppo algorithm
        ref_model (Critic): the reference model in ppo algorithm
        booster (Strategy): the strategy to use for training
        actor_optim (Optimizer): the optimizer to use for actor model
        actor_lr_scheduler (_LRScheduler): the lr scheduler to use for actor model
        tokenizer (PreTrainedTokenizerBase): the tokenizer to use for encoding
        max_epochs (int, defaults to 1): the max number of epochs to train
        accumulation_steps (int): the number of steps to accumulate gradients
        start_epoch (int, defaults to 0): the start epoch, non-zero if resumed from a checkpoint
        save_interval (int): the interval to save model checkpoints, default to 0, which means no checkpoint will be saved during trainning
        save_dir (str): the directory to save checkpoints
        coordinator (DistCoordinator): the coordinator to use for distributed logging
        beta (float, defaults to 0.1): the beta parameter in kto loss
        desirable_weight (float, defaults to 1.0): the weight for desirable reward
        undesirable_weight (float, defaults to 1.0): the weight for undesirable reward
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
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
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
        self.kto_loss = KTOLoss(beta=beta, desirable_weight=desirable_weight, undesirable_weight=undesirable_weight)
        self.apply_loss_mask = apply_loss_mask
        self.save_interval = save_interval
        self.coordinator = coordinator
        self.save_dir = save_dir
        self.num_train_step = 0
        self.accumulation_steps = accumulation_steps
        self.device = get_current_device()
        self.accumulative_meter = AccumulativeMeanMeter()
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
        self.beta = beta

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

            self.wandb_run = wandb.init(project="Coati-kto", sync_tensorboard=True)
        if log_dir is not None and is_rank_0():
            import os
            import time

            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(log_dir, "kto")
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
            (input_ids, attention_mask, loss_mask, label, kl_input_ids, kl_attention_mask, kl_loss_mask) = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["loss_mask"],
                batch["label"],
                batch["kl_input_ids"],
                batch["kl_attention_mask"],
                batch["kl_loss_mask"],
            )
            if not self.apply_loss_mask:
                loss_mask = loss_mask.fill_(1.0)
                kl_loss_mask = kl_loss_mask.fill_(1.0)

            batch_size = input_ids.size()[0]

            # actor logits
            with torch.no_grad():
                # calculate KL term with KT data
                kl_logits = self.model(
                    input_ids=kl_input_ids,
                    attention_mask=kl_attention_mask,
                )["logits"]

            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["logits"]

            logprob = calc_masked_log_probs(logits, input_ids, loss_mask[:, 1:]).sum(-1)
            kl_logprob = calc_masked_log_probs(kl_logits, kl_input_ids, kl_loss_mask[:, 1:]).sum(-1)
            chosen_index = [i for i in range(batch_size) if label[i] == 1]
            rejected_index = [i for i in range(batch_size) if label[i] == 0]
            chosen_logprob = logprob[chosen_index]
            rejected_logprob = logprob[rejected_index]
            with torch.no_grad():
                ref_kl_logits = self.ref_model(
                    input_ids=kl_input_ids,
                    attention_mask=kl_attention_mask,
                )["logits"]
                ref_logits = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )["logits"]

            ref_logprob = calc_masked_log_probs(ref_logits, input_ids, loss_mask[:, 1:]).sum(-1)
            ref_kl_logprob = calc_masked_log_probs(ref_kl_logits, kl_input_ids, kl_loss_mask[:, 1:]).sum(-1)
            ref_chosen_logprob = ref_logprob[chosen_index]
            ref_rejected_logprob = ref_logprob[rejected_index]

            loss, chosen_rewards, rejected_rewards, kl = self.kto_loss(
                chosen_logprob, rejected_logprob, kl_logprob, ref_chosen_logprob, ref_rejected_logprob, ref_kl_logprob
            )

            self.booster.backward(loss=loss, optimizer=self.optimizer)
            if self.num_train_step % self.accumulation_steps == self.accumulation_steps - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.actor_scheduler.step()

            # sync
            loss_mean = all_reduce_mean(tensor=loss)
            chosen_reward_mean = chosen_rewards.mean()
            chosen_rewards_list = [
                torch.tensor(0, dtype=loss.dtype, device=loss.device) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(chosen_rewards_list, chosen_reward_mean)
            rejected_reward_mean = rejected_rewards.mean()
            rejected_rewards_list = [
                torch.tensor(0, dtype=loss.dtype, device=loss.device) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(rejected_rewards_list, rejected_reward_mean)
            chosen_rewards_list = [i for i in chosen_rewards_list if not i.isnan()]
            rejected_rewards_list = [i for i in rejected_rewards_list if not i.isnan()]
            chosen_rewards_mean = (
                torch.stack(chosen_rewards_list).mean()
                if len(chosen_rewards_list) > 0
                else torch.tensor(torch.nan, dtype=loss.dtype, device=loss.device)
            )
            rejected_rewards_mean = (
                torch.stack(rejected_rewards_list).mean()
                if len(rejected_rewards_list) > 0
                else torch.tensor(torch.nan, dtype=loss.dtype, device=loss.device)
            )
            self.accumulative_meter.add("chosen_rewards", chosen_rewards_mean.to(torch.float16).mean().item())
            self.accumulative_meter.add("rejected_rewards", rejected_rewards_mean.to(torch.float16).mean().item())
            self.accumulative_meter.add("loss", loss_mean.to(torch.float16).detach().item())

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
        self.accumulative_meter.reset()
        step_bar = trange(
            len(self.train_dataloader) // self.accumulation_steps,
            desc=f"Epoch {epoch + 1}/{self.max_epochs}",
            disable=not is_rank_0(),
        )
        for i, batch in enumerate(self.train_dataloader):
            batch = to_device(batch, self.device)
            (input_ids, attention_mask, loss_mask, label, kl_input_ids, kl_attention_mask, kl_loss_mask) = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["loss_mask"],
                batch["label"],
                batch["kl_input_ids"],
                batch["kl_attention_mask"],
                batch["kl_loss_mask"],
            )

            if not self.apply_loss_mask:
                loss_mask = loss_mask.fill_(1.0)
                kl_loss_mask = kl_loss_mask.fill_(1.0)

            batch_size = input_ids.size()[0]

            # actor logits
            with torch.no_grad():
                # calculate KL term with KT data
                kl_logits = self.model(
                    input_ids=kl_input_ids,
                    attention_mask=kl_attention_mask,
                )["logits"]

                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )["logits"]

            logprob = calc_masked_log_probs(logits, input_ids, loss_mask[:, 1:]).sum(-1)
            kl_logprob = calc_masked_log_probs(kl_logits, kl_input_ids, kl_loss_mask[:, 1:]).sum(-1)
            chosen_index = [i for i in range(batch_size) if label[i] == 1]
            rejected_index = [i for i in range(batch_size) if label[i] == 0]
            chosen_logprob = logprob[chosen_index]
            rejected_logprob = logprob[rejected_index]
            with torch.no_grad():
                ref_kl_logits = self.ref_model(
                    input_ids=kl_input_ids,
                    attention_mask=kl_attention_mask,
                )["logits"]

                ref_logits = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )["logits"]

            ref_logprob = calc_masked_log_probs(ref_logits, input_ids, loss_mask[:, 1:]).sum(-1)
            ref_kl_logprob = calc_masked_log_probs(ref_kl_logits, kl_input_ids, kl_loss_mask[:, 1:]).sum(-1)
            ref_chosen_logprob = ref_logprob[chosen_index]
            ref_rejected_logprob = ref_logprob[rejected_index]

            loss, chosen_rewards, rejected_rewards, kl = self.kto_loss(
                chosen_logprob, rejected_logprob, kl_logprob, ref_chosen_logprob, ref_rejected_logprob, ref_kl_logprob
            )

            # sync
            loss_mean = all_reduce_mean(tensor=loss)
            chosen_rewards_mean = all_reduce_mean(tensor=chosen_rewards.mean())
            rejected_rewards_mean = all_reduce_mean(tensor=rejected_rewards.mean())
            self.accumulative_meter.add("chosen_rewards", chosen_rewards_mean.to(torch.float16).mean().item())
            self.accumulative_meter.add("rejected_rewards", rejected_rewards_mean.to(torch.float16).mean().item())
            self.accumulative_meter.add("loss", loss_mean.to(torch.float16).detach().item())
            self.accumulative_meter.add(
                "margin", (chosen_rewards_mean - rejected_rewards_mean).to(torch.float16).mean().item()
            )
            step_bar.update()
        msg = "Evaluation Result:\n"
        for tag in ["loss", "chosen_rewards", "rejected_rewards", "margin"]:
            msg = msg + f"{tag}: {self.accumulative_meter.get(tag)}\n"
        self.coordinator.print_on_master(msg)
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, f"eval_result_epoch{epoch}.txt"), "w") as f:
            f.write(msg)
        step_bar.close()
