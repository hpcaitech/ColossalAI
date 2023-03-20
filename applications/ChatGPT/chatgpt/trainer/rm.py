from abc import ABC
import pandas as pd
import loralib as lora
import torch
from datetime import datetime
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
 
from .strategies import Strategy
from .utils import is_rank_0


class RewardModelTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        loss_fn (callable): the loss function to use for training
        train_dataset (Dataset): the dataset to use for training
        valid_dataset (Dataset): the dataset to use for validation
        eval_dataset (Dataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
    """

    def __init__(
        self,
        model,
        strategy: Strategy,
        optim: Optimizer,
        loss_fn,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        eval_dataset: Dataset,
        batch_size: int = 1,
        max_epochs: int = 1,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
        
        self.model = strategy.setup_model(model)
        self.loss_fn = loss_fn
        self.optimizer = strategy.setup_optimizer(optim, self.model)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, self.train_dataloader.__len__()//100)


    def eval_acc(self, dataloader):
        dist = 0
        on = 0
        cnt = 0
        self.model.eval()
        with torch.no_grad():
            for chosen_ids, c_mask, reject_ids, r_mask in dataloader:
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
            dist_mean = dist / len(dataloader)
            acc = on / cnt
        self.model.train()
        return dist_mean, acc
    

    def fit(self):
        time = datetime.now()
        epoch_bar = tqdm(range(self.epochs), desc='Train epoch', disable=not is_rank_0())
        for epoch in range(self.epochs):
            step_bar = tqdm(range(self.train_dataloader.__len__()),
                            desc='Train step of epoch %d' % epoch,
                            disable=not is_rank_0())
            # train
            self.model.train()
            cnt = 0
            acc = 0
            dist = 0
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
                cnt += 1
                if cnt == 100:
                    self.scheduler.step()
                    dist, acc = self.eval_acc(self.valid_dataloader)
                    cnt = 0
                    if is_rank_0():
                        log = pd.DataFrame([[step_bar.n, loss.item(), dist, acc]], columns=['step', 'loss', 'dist', 'acc'])
                        log.to_csv('log_%s.csv' % time, mode='a', header=False, index=False)
                step_bar.update()
                step_bar.set_postfix({'dist': dist, 'acc': acc})
                
            # eval
            dist, acc = self.eval_acc(self.eval_dataloader)
            if is_rank_0():
                    log = pd.DataFrame([[step_bar.n, loss.item(), dist, acc]], columns=['step', 'loss', 'dist', 'acc'])
                    log.to_csv('log.csv', mode='a', header=False, index=False)
            epoch_bar.update()
            step_bar.set_postfix({'dist': dist, 'acc': acc})
            step_bar.close()
