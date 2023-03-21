from abc import ABC
from typing import Optional
import loralib as lora
import torch
from chatgpt.dataset import SFTDataset
from chatgpt.models.loss import GPTLMLoss
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
from .strategies import Strategy
from .utils import is_rank_0
from colossalai.logging import get_dist_logger


class SFTTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (SFTDataset or SFTDistributedDataset): the dataset to use for training
        eval_dataset (SFTDataset or SFTDistributedDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        strategy: Strategy,
        optim: Optimizer,
        train_dataset: SFTDataset,
        eval_dataset: SFTDataset,
        sampler: Optional[DistributedSampler] = None,
        batch_size: int = 1,
        max_epochs: int = 2,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.sampler = sampler

        self.train_dataloader = DataLoader(self.train_dataset, shuffle=(sampler is None),
                                           sampler=sampler, batch_size=batch_size)
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=batch_size)

        self.model = strategy.setup_model(model)
        if "DDP" in str(self.strategy):
            self.model = self.model.module
        self.loss_fn = GPTLMLoss()
        self.optimizer = strategy.setup_optimizer(optim, self.model)

    def fit(self, logger, use_lora, log_interval=10):
        epoch_bar = tqdm(range(self.epochs), desc='Train epoch', disable=not is_rank_0())
        for epoch in range(self.epochs):
            if isinstance(self.sampler, DistributedSampler):
                self.sampler.set_epoch(epoch)
            # train
            self.model.train()
            for batch_id, batch in enumerate(self.train_dataloader):
                prompt_ids = batch["input_ids"]
                p_mask = batch["attention_mask"]
                prompt_ids = prompt_ids.squeeze(1).cuda()
                p_mask = p_mask.squeeze(1).cuda()
                prompt_logits = self.model(prompt_ids, attention_mask=p_mask)

                loss = self.loss_fn(prompt_logits, prompt_ids)
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer)
                self.optimizer.zero_grad()
                if batch_id % log_interval == 0:
                    logger.info(f'Train Epoch {epoch}/{self.epochs} Batch {batch_id} Rank {dist.get_rank()} loss {loss.item()}')

            # eval
            self.model.eval()
            with torch.no_grad():
                loss_sum = 0
                num_seen = 0
                for batch in self.eval_dataloader:
                    prompt_ids = batch["input_ids"]
                    p_mask = batch["attention_mask"]
                    prompt_ids = prompt_ids.squeeze(1).cuda()
                    p_mask = p_mask.squeeze(1).cuda()

                    prompt_logits = self.model(prompt_ids, attention_mask=p_mask)
                    loss = self.loss_fn(prompt_logits, prompt_ids)
                    loss_sum += loss.item()
                    num_seen += prompt_ids.size(0)

                loss_mean = loss_sum / num_seen
                if dist.get_rank() == 0:
                    logger.info(f'Eval Epoch {epoch}/{self.epochs} loss {loss_mean}')
            epoch_bar.update()

