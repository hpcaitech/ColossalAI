"""
Trainer for Process Reward Model.
"""

import os
import time
from typing import Any, Callable, List, Optional

import torch
import tqdm
from coati.models import PRMLoss
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


class ProcessRewardModelTrainer(SLTrainer):
    """
    Trainer for Process Reward Model.
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
        accumulation_steps: int = 1,
        start_epoch: int = 0,
        save_interval: int = 0,
        save_dir: str = None,
        coordinator: DistCoordinator = None,
        reward_signal_ids: List[int] = [],
    ) -> None:
        super().__init__(
            booster, max_epochs=max_epochs, model=model, optimizer=optimizer, plugin=plugin, start_epoch=start_epoch
        )
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer
        self.reward_signal_ids = reward_signal_ids
        self.loss_fn = loss_fn if loss_fn is not None else PRMLoss(self.reward_signal_ids)
        self.save_interval = save_interval
        self.coordinator = coordinator
        self.save_dir = save_dir
        self.num_train_step = 0
        self.accumulation_steps = accumulation_steps
        self.device = get_current_device()
        self.accumulative_meter = AccumulativeMeanMeter()

    def _before_fit(
        self,
        train_dataloader: DataLoader = None,
        eval_dataloader: DataLoader = None,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
    ):
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.writer = None
        if log_dir is not None and is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(log_dir, "PRM", time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
            self.writer = SummaryWriter(log_dir=log_dir)

            if use_wandb:
                import wandb

                self.wandb_run = wandb.init(project="Coati-PRM", sync_tensorboard=True)

    def _train(self, epoch: int):
        self.model.train()
        step_bar = tqdm.trange(
            len(self.train_dataloader) // self.accumulation_steps,
            desc=f"Epoch {epoch + 1}/{self.max_epochs}",
            disable=not is_rank_0(),
        )
        for i, batch in enumerate(self.train_dataloader):
            batch = to_device(batch, self.device)
            batch_size = batch["input_ids"].size(0)
            logits = self.model(batch["input_ids"])["logits"]
            loss = self.loss_fn(batch["labels"], logits)
            self.booster.backward(loss=loss, optimizer=self.optimizer)
            loss_mean = all_reduce_mean(tensor=loss)
            self.accumulative_meter.add("loss", loss_mean.to(torch.float16).item())

            if (i + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                step_bar.set_postfix({"train/loss": self.accumulative_meter.get("loss")})
                if self.writer:
                    self.writer.add_scalar("train/loss", self.accumulative_meter.get("loss"), self.num_train_step)
                self.num_train_step += 1
                step_bar.update()

            # Save checkpoint
            if (
                self.save_dir is not None
                and self.save_interval is not None
                and (self.num_train_step + 1) % self.save_interval == 0
            ):
                save_checkpoint(
                    save_dir=self.save_dir,
                    booster=self.booster,
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.scheduler,
                    epoch=epoch,
                    step=self.num_train_step + 1,
                    batch_size=batch_size,
                    coordinator=self.coordinator,
                )
                self.coordinator.print_on_master(
                    f"Saved checkpoint at epoch {epoch} step {self.num_train_step} at folder {self.save_dir}"
                )

    def _eval(self, epoch: int):
        self.model.eval()

        step_bar = tqdm.trange(
            len(self.eval_dataloader),
            desc=f"Epoch {epoch + 1}/{self.max_epochs}",
            disable=not is_rank_0(),
        )
        for batch in self.eval_dataloader:
            batch = to_device(batch, self.device)
            logits = self.model(batch["input_ids"])["logits"]
            loss = self.loss_fn(batch["labels"], logits)
            loss_mean = all_reduce_mean(tensor=loss)
            self.accumulative_meter.add(
                "loss", loss_mean.to(torch.float16).item(), count_update=batch["input_ids"].size(0)
            )
            step_bar.update()

        loss_mean = self.accumulative_meter.get("loss")
        msg = "Evaluation Result:\n"
        for tag in ["loss"]:
            msg = msg + f"{tag}: {self.accumulative_meter.get(tag)}\n"
        self.coordinator.print_on_master(msg)
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, f"eval_result_epoch{epoch}.txt"), "w") as f:
                f.write(msg)
        step_bar.close()
