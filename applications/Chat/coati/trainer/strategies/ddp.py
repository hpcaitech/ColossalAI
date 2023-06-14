import os
import random
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from coati.replay_buffer import ReplayBuffer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .naive import NaiveStrategy
from .sampler import DistributedSampler


class DDPStrategy(NaiveStrategy):
    """
        Strategy for distributed training using torch.distributed.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        super().__init__()

    def setup_distributed(self) -> None:
        self._try_init_dist(force=True)
        self.set_seed(self.seed)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def setup_model(self, model: nn.Module) -> nn.Module:
        device = torch.cuda.current_device()
        return DDP(model, device_ids=[device])

    def setup_dataloader(self, replay_buffer: ReplayBuffer, pin_memory: bool = False) -> DataLoader:
        # DDP only mode, replay buffers on each rank are different.
        # sampler = DistributedSampler(replay_buffer,
        #                              num_replicas=dist.get_world_size(),
        #                              rank=dist.get_rank(),
        #                              shuffle=True,
        #                              seed=self.seed,
        #                              drop_last=True)
        return DataLoader(
            replay_buffer,
            batch_size=replay_buffer.sample_batch_size,
        #   sampler=sampler,
            shuffle=True,
            drop_last=True,
            pin_memory=pin_memory,
            collate_fn=replay_buffer.collate_fn)

    def save_model(self, model: nn.Module, path: str, only_rank0: bool = True) -> None:
        if only_rank0 and dist.get_rank() != 0:
            return
        super().save_model(model, path, only_rank0)

    def save_optimizer(self, optimizer: Optimizer, path: str, only_rank0: bool = False) -> None:
        if only_rank0 and dist.get_rank() != 0:
            return
        super().save_optimizer(optimizer, path, only_rank0)

    def setup_sampler(self, dataset) -> DistributedSampler:
        return DistributedSampler(dataset, dist.get_world_size(), dist.get_rank())

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        assert isinstance(model, DDP)
        return model.module

    def save_pretrained(self,
                        model: nn.Module,
                        path: str,
                        only_rank0: bool = True,
                        tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        if only_rank0 and dist.get_rank() != 0:
            return
        super().save_pretrained(model, path, only_rank0, tokenizer)
