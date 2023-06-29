import time
from typing import Optional

import torch
import torch.distributed as dist
import tqdm
import wandb
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
            assert not isinstance(strategy, GeminiStrategy), \
                "Accumulation steps are not supported in stage 3 of ColossalAI"

        super().__init__(strategy, max_epochs, model, optim)

        self.accumulation_steps = accumulation_steps
        self.scheduler = lr_scheduler

    def _train(self, epoch: int):
        self.model.train()
        for batch_id, batch in enumerate(self.train_dataloader):

            batch = to_device(batch, torch.cuda.current_device())
            outputs = self.model(batch["input_ids"],
                                 attention_mask=batch["attention_mask"],
                                 labels=batch["labels"])

            loss = outputs.loss
            loss = loss / self.accumulation_steps

            self.strategy.backward(loss, self.model, self.optimizer)

            self.total_loss += loss.item()

            # gradient accumulation
            if (batch_id + 1) % self.accumulation_steps == 0:
                self.strategy.optimizer_step(self.optimizer)
                self.optimizer.zero_grad()
                self.scheduler.step()
                if is_rank_0() and self.use_wandb:
                    wandb.log({
                        "loss": self.total_loss / self.accumulation_steps,
                        "lr": self.scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "batch_id": batch_id
                    })
                self.total_loss = 0
                self.step_bar.update()

    def _eval(self, epoch: int):
        if self.eval_dataloader is not None:
            self.model.eval()
            with torch.no_grad():
                loss_sum, num_seen = 0, 0
                for batch in self.eval_dataloader:
                    batch = to_device(batch, torch.cuda.current_device())
                    outputs = self.model(batch["input_ids"],
                                         attention_mask=batch["attention_mask"],
                                         labels=batch["labels"])
                    loss = outputs.loss

                    loss_sum += loss.item()
                    num_seen += batch["input_ids"].size(0)

                loss_mean = loss_sum / num_seen
                if dist.get_rank() == 0:
                    self.logger.info(f'Eval Epoch {epoch}/{self.max_epochs} loss {loss_mean}')

    def _before_fit(self,
                    train_dataloader: DataLoader,
                    eval_dataloader: Optional[DataLoader] = None,
                    logger: Optional[DistributedLogger] = None,
                    use_wandb: bool = False):
        """
        Args:
            train_dataloader: the dataloader to use for training
            eval_dataloader: the dataloader to use for evaluation
        """
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.logger = logger
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="Coati", name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            wandb.watch(self.model)

        self.total_loss = 0
        self.no_epoch_bar = True
        self.step_bar = tqdm.trange(
            len(self.train_dataloader) // self.accumulation_steps * self.max_epochs,
            desc=f'steps',
            disable=not is_rank_0()
        )
