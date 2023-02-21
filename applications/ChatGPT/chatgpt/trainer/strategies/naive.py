from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from chatgpt.replay_buffer import ReplayBuffer
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .base import Strategy


class NaiveStrategy(Strategy):
    """
        Strategy for single GPU. No parallelism is used.
    """

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        loss.backward()

    def optimizer_step(self, optimizer: optim.Optimizer, **kwargs) -> None:
        optimizer.step()

    def setup_distributed(self) -> None:
        pass

    def setup_model(self, model: nn.Module) -> nn.Module:
        return model

    def setup_optimizer(self, optimizer: optim.Optimizer, model: nn.Module) -> optim.Optimizer:
        return optimizer

    def setup_dataloader(self, replay_buffer: ReplayBuffer, pin_memory: bool = False) -> DataLoader:
        return DataLoader(replay_buffer,
                          batch_size=replay_buffer.sample_batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=pin_memory,
                          collate_fn=replay_buffer.collate_fn)

    def save_model(self, model: nn.Module, path: str, only_rank0: bool = False) -> None:
        unwrapped_model = self._unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), path)

    def load_model(self, model: nn.Module, path: str, map_location: Any = None, strict: bool = True) -> None:
        unwrapped_model = self._unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    def save_optimizer(self, optimizer: Optimizer, path: str, only_rank0: bool = False) -> None:
        torch.save(optimizer.state_dict(), path)

    def load_optimizer(self, optimizer: Optimizer, path: str, map_location: Any = None) -> None:
        state_dict = torch.load(path, map_location=map_location)
        optimizer.load_state_dict(state_dict)
