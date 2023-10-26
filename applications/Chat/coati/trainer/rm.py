from typing import Callable, Optional

import torch
import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from .base import SLTrainer
from .strategies import Strategy
from .utils import is_rank_0


class RewardModelTrainer(SLTrainer):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim (Optimizer): the optimizer to use for training
        lr_scheduler (_LRScheduler): the lr scheduler to use for training
        loss_fn (callable): the loss function to use for training
        max_epochs (int, defaults to 2): the number of epochs to train
    """

    def __init__(
        self,
        model,
        strategy: Strategy,
        optim: Optimizer,
        lr_scheduler: _LRScheduler,
        loss_fn: Callable,
        max_epochs: int = 1,
    ) -> None:
        super().__init__(strategy, max_epochs, model, optim)

        self.loss_fn = loss_fn
        self.scheduler = lr_scheduler

        self.num_train_step = 0

    def _eval(self, epoch):
        if self.eval_dataloader is not None:
            self.model.eval()
            dist, num_correct, num_samples = 0, 0, 0
            mean_reward_choen, mean_reward_reject = [], []
            with torch.no_grad():
                for chosen_ids, c_mask, reject_ids, r_mask in self.eval_dataloader:
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                    chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                    reject_reward = self.model(reject_ids, attention_mask=r_mask)
                    mean_reward_choen.append(chosen_reward.mean().item())
                    mean_reward_reject.append(reject_reward.mean().item())
                    num_samples += chosen_ids.size(0)
                    num_correct += (chosen_reward > reject_reward).sum().item()
                    dist += (chosen_reward - reject_reward).mean().item()

                self.dist = dist / len(self.eval_dataloader)
                self.acc = num_correct / num_samples

            if self.writer:
                self.writer.add_scalar("eval/mean_reward_choen", sum(mean_reward_choen) / len(mean_reward_choen), epoch)
                self.writer.add_scalar(
                    "eval/mean_reward_reject", sum(mean_reward_reject) / len(mean_reward_reject), epoch
                )
                self.writer.add_scalar("eval/dist", self.dist, epoch)
                self.writer.add_scalar("eval/acc", self.acc, epoch)

    def _train(self, epoch):
        self.model.train()
        step_bar = tqdm.trange(
            len(self.train_dataloader), desc=f"Epoch {epoch + 1}/{self.max_epochs}", disable=not is_rank_0()
        )
        for chosen_ids, c_mask, reject_ids, r_mask in self.train_dataloader:
            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
            c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
            r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
            chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
            reject_reward = self.model(reject_ids, attention_mask=r_mask)
            loss = self.loss_fn(chosen_reward, reject_reward)
            self.strategy.backward(loss, self.model, self.optimizer)
            self.strategy.optimizer_step(self.optimizer)
            self.optimizer.zero_grad()
            if self.writer:
                self.writer.add_scalar("train/loss", loss.item(), self.num_train_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.num_train_step)
                self.writer.add_scalar("train/dist", (chosen_reward - reject_reward).mean().item(), self.num_train_step)
                self.writer.add_scalar("train/reward_chosen", chosen_reward.mean().item(), self.num_train_step)
                self.writer.add_scalar("train/reward_reject", reject_reward.mean().item(), self.num_train_step)
                self.writer.add_scalar(
                    "train/acc", (chosen_reward > reject_reward).float().mean().item(), self.num_train_step
                )
            self.num_train_step += 1
            if self.num_train_step % 100 == 0:
                self.scheduler.step()
            step_bar.update()
        step_bar.close()

    def _before_fit(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
    ):
        """
        Args:
            train_dataloader (DataLoader): the dataloader to use for training
            eval_dataloader (DataLoader): the dataloader to use for evaluation
        """
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.writer = None
        if use_wandb and is_rank_0():
            assert log_dir is not None, "log_dir must be provided when use_wandb is True"
            import wandb

            wandb.init(project="Coati-rm", sync_tensorboard=True)
        if log_dir is not None and is_rank_0():
            import os
            import time

            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(log_dir, "rm")
            log_dir = os.path.join(log_dir, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
            self.writer = SummaryWriter(log_dir=log_dir)
