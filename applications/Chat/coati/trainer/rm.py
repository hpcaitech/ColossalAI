from datetime import datetime
from typing import Callable

import pandas as pd
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

    def _eval(self, epoch):
        if self.eval_dataloader is not None:
            self.model.eval()
            dist, on, cnt = 0, 0, 0
            with torch.no_grad():
                for chosen_ids, c_mask, reject_ids, r_mask in self.eval_dataloader:
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                    chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                    reject_reward = self.model(reject_ids, attention_mask=r_mask)
                    for i in range(len(chosen_reward)):
                        cnt += 1
                        if chosen_reward[i] > reject_reward[i]:
                            on += 1
                    dist += (chosen_reward - reject_reward).mean().item()
                self.dist = dist / len(self.eval_dataloader)
                self.acc = on / cnt

            if is_rank_0():
                log = pd.DataFrame(
                    [[(epoch + 1) * len(self.train_dataloader),
                      self.loss.item(), self.dist, self.acc]],
                    columns=['step', 'loss', 'dist', 'acc']
                )
                log.to_csv('log.csv', mode='a', header=False, index=False)

    def _train(self, epoch):
        self.model.train()
        step_bar = tqdm.trange(
            len(self.train_dataloader),
            desc='Train step of epoch %d' % epoch,
            disable=not is_rank_0()
        )
        cnt = 0
        for chosen_ids, c_mask, reject_ids, r_mask in self.train_dataloader:
            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
            c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
            r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
            chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
            reject_reward = self.model(reject_ids, attention_mask=r_mask)
            self.loss = self.loss_fn(chosen_reward, reject_reward)
            self.strategy.backward(self.loss, self.model, self.optimizer)
            self.strategy.optimizer_step(self.optimizer)
            self.optimizer.zero_grad()
            cnt += 1
            if cnt % 100 == 0:
                self.scheduler.step()
            step_bar.update()
        step_bar.close()

    def _before_fit(self,
                    train_dataloader: DataLoader,
                    valid_dataloader: DataLoader,
                    eval_dataloader: DataLoader):
        """
        Args:
            train_dataloader (DataLoader): the dataloader to use for training
            valid_dataloader (DataLoader): the dataloader to use for validation
            eval_dataloader (DataLoader): the dataloader to use for evaluation
        """
        super()._before_fit()
        self.datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.eval_dataloader = eval_dataloader
