import torch
import torch.nn as nn
import torch.optim as optim
from chatgpt.replay_buffer import ReplayBuffer
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
