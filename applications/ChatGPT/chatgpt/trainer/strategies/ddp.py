import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from chatgpt.nn import Actor
from chatgpt.replay_buffer import ReplayBuffer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from .base import Strategy
from .naive import NaiveStrategy


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
        sampler = DistributedSampler(replay_buffer,
                                     num_replicas=dist.get_world_size(),
                                     rank=dist.get_rank(),
                                     shuffle=True,
                                     seed=self.seed,
                                     drop_last=True)
        return DataLoader(replay_buffer,
                          batch_size=replay_buffer.sample_batch_size,
                          sampler=sampler,
                          pin_memory=pin_memory,
                          collate_fn=replay_buffer.collate_fn)

    @staticmethod
    def _unwrap_actor(actor: Actor) -> nn.Module:
        model: DDP = Strategy._unwrap_actor(actor)
        return model.module

    def save_model(self, model: nn.Module, path: str, only_rank0: bool = False) -> None:
        if only_rank0 and dist.get_rank() != 0:
            return
        super().save_model(model, path, only_rank0)

    def save_optimizer(self, optimizer: Optimizer, path: str, only_rank0: bool = False) -> None:
        if only_rank0 and dist.get_rank() != 0:
            return
        super().save_optimizer(optimizer, path, only_rank0)
