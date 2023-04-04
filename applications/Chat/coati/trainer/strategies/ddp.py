from typing import Optional

import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from coati.models.base import LM, Actor, RewardModel
from coati.models.lora import LoraLinear
from coati.replay_buffer import ReplayBuffer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .base import Strategy
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
        try:
            rank = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            host = os.environ['MASTER_ADDR']
            port = int(os.environ['MASTER_PORT'])
        except KeyError as e:
            raise RuntimeError(
                f"Could not find {e} in the torch environment, visit https://www.colossalai.org/ for more information on launching with torch"
            )
        dist.init_process_group('nccl', init_method=f'tcp://[{host}]:{port}', world_size=world_size, rank=rank)
        self.set_seed(self.seed)
        torch.cuda.set_device(local_rank)

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

    @staticmethod
    def _unwrap_actor(actor: Actor) -> nn.Module:
        model: DDP = Strategy._unwrap_actor(actor)
        return model.module

    def save_model(self, model: nn.Module, path: str, only_rank0: bool = False, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        if only_rank0 and dist.get_rank() != 0:
            return None
        
        for module in model.modules():
            if isinstance(module, LoraLinear):
                module.merge_weights = True
                module.eval()
        
        if isinstance(model, RewardModel):
            state_dict = model.state_dict()
            if only_rank0 and dist.get_rank() != 0:
                return
            torch.save(state_dict, path)
        else:
            try:
                if isinstance(model, LM):
                    model = model.model
                model.save_pretrained(path)
                if tokenizer is not None:
                    tokenizer.save_pretrained(path)
            except AttributeError:
                state_dict = model.state_dict()
                if only_rank0 and dist.get_rank() != 0:
                    return
                torch.save(state_dict, path)

    def save_optimizer(self, optimizer: Optimizer, path: str, only_rank0: bool = False) -> None:
        if only_rank0 and dist.get_rank() != 0:
            return
        super().save_optimizer(optimizer, path, only_rank0)

    def setup_sampler(self, dataset) -> DistributedSampler:
        return DistributedSampler(dataset, dist.get_world_size(), dist.get_rank())
