from typing import Optional

import torch
import torch.distributed as dist
import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from colossalai.logging import DistributedLogger

from .base import SLTrainer
from .strategies import GeminiStrategy, Strategy
from .utils import is_rank_0, to_device


class SFTTrainer(SLTrainer):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        lr_scheduler(_LRScheduler): the lr scheduler to use for training
        max_epochs (int, defaults to 2): the number of epochs to train
        accumulation_steps (int, defaults to 8): the number of steps to accumulate gradients
    """

    def __init__(
        self,
        model,
        strategy: Strategy,
        optim: Optimizer,
        lr_scheduler: _LRScheduler,
        max_epochs: int = 2,
        accumulation_steps: int = 8,
    ) -> None:
        if accumulation_steps > 1:
            assert not isinstance(
                strategy, GeminiStrategy
            ), "Accumulation steps are not supported in stage 3 of ColossalAI"

        super().__init__(strategy, max_epochs, model, optim)

        self.accumulation_steps = accumulation_steps
        self.scheduler = lr_scheduler

        self.num_train_step = 0
        self.num_eval_step = 0

    def _train(self, epoch: int):
        self.model.train()
        step_bar = tqdm.trange(
            len(self.train_dataloader) // self.accumulation_steps,
            desc=f"Epoch {epoch + 1}/{self.max_epochs}",
            disable=not is_rank_0(),
        )
        for i, batch in enumerate(self.train_dataloader):
            batch = to_device(batch, torch.cuda.current_device())
            outputs = self.model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = outputs.loss / self.accumulation_steps
            self.total_loss += loss.item()
            self.strategy.backward(loss, self.model, self.optimizer)
            # gradient accumulation
            if (i + 1) % self.accumulation_steps == 0:
                self.strategy.optimizer_step(self.optimizer)
                self.optimizer.zero_grad()
                self.scheduler.step()
                if self.writer:
                    self.writer.add_scalar("train/loss", self.total_loss, self.num_train_step)
                    self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.num_train_step)
                    self.num_train_step += 1
                self.total_loss = 0
                step_bar.update()
        step_bar.close()

    def _eval(self, epoch: int):
        if self.eval_dataloader is not None:
            self.model.eval()
            with torch.no_grad():
                loss_sum, num_seen = 0, 0
                for batch in self.eval_dataloader:
                    batch = to_device(batch, torch.cuda.current_device())
                    outputs = self.model(
                        batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]
                    )
                    loss_sum += outputs.loss.item()
                    num_seen += batch["input_ids"].size(0)
                loss_mean = loss_sum / num_seen
                if dist.get_rank() == 0:
                    self.logger.info(f"Eval Epoch {epoch}/{self.max_epochs} loss {loss_mean}")
                if self.writer:
                    self.writer.add_scalar("eval/loss", loss_mean, self.num_eval_step)
                    self.num_eval_step += 1

    def _before_fit(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        logger: Optional[DistributedLogger] = None,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
    ):
        """
        Args:
            train_dataloader: the dataloader to use for training
            eval_dataloader: the dataloader to use for evaluation
        """
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.logger = logger
        self.writer = None
        if use_wandb and is_rank_0():
            assert log_dir is not None, "log_dir must be provided when use_wandb is True"
            import wandb

            wandb.init(project="Coati-sft", sync_tensorboard=True)
        if log_dir is not None and is_rank_0():
            import os
            import time

            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(log_dir, "sft")
            log_dir = os.path.join(log_dir, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
            self.writer = SummaryWriter(log_dir=log_dir)

        self.total_loss = 0
