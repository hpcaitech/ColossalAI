import random
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from coati.replay_buffer import ReplayBuffer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from colossalai.booster.plugin import TorchDDPPlugin  # This is a wrapper of torch.DDP
from colossalai.booster.plugin.torch_ddp_plugin import TorchDDPModel

from .naive import NaiveStrategy
from .sampler import DistributedSampler


class DDPStrategy(NaiveStrategy):
    """
        Strategy for distributed training using torch.distributed.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        super().__init__(plugin_initializer=TorchDDPPlugin)

    def setup_distributed(self) -> None:
        self._try_init_dist(force=True)
        self.set_seed(self.seed)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def setup_dataloader(self, replay_buffer: ReplayBuffer, pin_memory: bool = False) -> DataLoader:
        # DDP only mode, replay buffers on each rank are different.
        assert isinstance(self.plugin, TorchDDPPlugin), "self.plugin isn't initialized properly."
        return self.plugin.prepare_dataloader(replay_buffer,
                                              batch_size=replay_buffer.sample_batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              pin_memory=pin_memory,
                                              collate_fn=replay_buffer.collate_fn)

    def setup_sampler(self, dataset) -> DistributedSampler:
        # FIXME(cwher): this is only invoked in train_on_ray, not tested after adapt Boost API.
        return DistributedSampler(dataset, dist.get_world_size(), dist.get_rank())

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        assert isinstance(model, TorchDDPModel), "model is not wrapped by TorchDDPModel."
        ddp_model = model.module
        assert isinstance(ddp_model, DDP), "TorchDDPModel should wrap a DDP model."
        return ddp_model.module

    def save_pretrained(self,
                        model: nn.Module,
                        path: str,
                        only_rank0: bool = True,
                        tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        if only_rank0 and dist.get_rank() != 0:
            return
        super().save_pretrained(model, path, only_rank0, tokenizer)
