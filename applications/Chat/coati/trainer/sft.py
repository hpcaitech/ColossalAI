import math
import time
from typing import Optional, List

import loralib as lora
import torch
import torch.distributed as dist
import wandb
from coati.models.loss import GPTLMLoss
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import get_scheduler

from colossalai.logging import get_dist_logger

from .callbacks import Callback
from .base import Trainer
from .strategies import Strategy
from .utils import is_rank_0


class SFTTrainer(Trainer):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataloader: the dataloader to use for training
        eval_dataloader: the dataloader to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        strategy: Strategy,
        optim: Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader = None,
        batch_size: int = 1,
        max_epochs: int = 2,
        accimulation_steps: int = 8,
        callbacks: List[Callback] = [],
    ) -> None:
        super().__init__(strategy, max_epochs, callbacks=callbacks)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.model = strategy.setup_model(model)
        if "DDP" in str(self.strategy):
            self.model = self.model.module
        self.optimizer = strategy.setup_optimizer(optim, self.model)

        self.accimulation_steps = accimulation_steps
        num_update_steps_per_epoch = len(train_dataloader) // self.accimulation_steps
        max_steps = math.ceil(self.max_epochs * num_update_steps_per_epoch)

        self.scheduler = get_scheduler("cosine",
                                       self.optimizer,
                                       num_warmup_steps=math.ceil(max_steps * 0.03),
                                       num_training_steps=max_steps)

    def fit(self, logger, log_interval=10):
        wandb.init(project="Coati", name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        wandb.watch(self.model)
        total_loss = 0
        # epoch_bar = tqdm(range(self.epochs), desc='Epochs', disable=not is_rank_0())
        step_bar = tqdm(range(len(self.train_dataloader) // self.accimulation_steps * self.max_epochs),
                        desc=f'steps',
                        disable=not is_rank_0())
        for epoch in range(self.max_epochs):

            # process_bar = tqdm(range(len(self.train_dataloader)), desc=f'Train process for{epoch}', disable=not is_rank_0())
            # train
            self.model.train()
            for batch_id, batch in enumerate(self.train_dataloader):

                prompt_ids = batch["input_ids"].to(torch.cuda.current_device())
                p_mask = batch["attention_mask"].to(torch.cuda.current_device())
                labels = batch["labels"].to(torch.cuda.current_device())
                # prompt_ids = prompt_ids.squeeze(1).cuda()
                # p_mask = p_mask.squeeze(1).cuda()
                # prompt_logits = self.model(prompt_ids, attention_mask=p_mask, labels=labels)

                outputs = self.model(prompt_ids, attention_mask=p_mask, labels=labels)

                loss = outputs.loss
                prompt_logits = outputs.logits

                if loss >= 2.5 and is_rank_0():
                    logger.warning(f"batch_id:{batch_id}, abnormal loss: {loss}")

                loss = loss / self.accimulation_steps

                self.strategy.backward(loss, self.model, self.optimizer)

                total_loss += loss.item()

                # gradient accumulation
                if (batch_id + 1) % self.accimulation_steps == 0:
                    self.strategy.optimizer_step(self.optimizer)
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    if is_rank_0():
                        wandb.log({
                            "loss": total_loss / self.accimulation_steps,
                            "lr": self.scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "batch_id": batch_id
                        })
                    total_loss = 0
                    step_bar.update()

                # if batch_id % log_interval == 0:
                # logger.info(f'Train Epoch {epoch}/{self.epochs} Batch {batch_id} Rank {dist.get_rank()} loss {loss.item()}')
                # wandb.log({"loss": loss.item()})

                # process_bar.update()

            # eval
            if self.eval_dataloader is not None:
                self.model.eval()
                with torch.no_grad():
                    loss_sum = 0
                    num_seen = 0
                    for batch in self.eval_dataloader:
                        prompt_ids = batch["input_ids"].to(torch.cuda.current_device())
                        p_mask = batch["attention_mask"].to(torch.cuda.current_device())
                        labels = batch["labels"].to(torch.cuda.current_device())
                        # prompt_ids = prompt_ids.squeeze(1).cuda()
                        # p_mask = p_mask.squeeze(1).cuda()

                        outputs = self.model(prompt_ids, attention_mask=p_mask, labels=labels)
                        loss = outputs.loss
                        # prompt_logits = outputs.logits

                        loss_sum += loss.item()
                        num_seen += prompt_ids.size(0)

                    loss_mean = loss_sum / num_seen
                    if dist.get_rank() == 0:
                        logger.info(f'Eval Epoch {epoch}/{self.max_epochs} loss {loss_mean}')

            # epoch_bar.update()

    def save_model(self,
                   path: str,
                   only_rank0: bool = False,
                   tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        self.strategy.save_model(model=self.model, path=path, only_rank0=only_rank0, tokenizer=tokenizer)
