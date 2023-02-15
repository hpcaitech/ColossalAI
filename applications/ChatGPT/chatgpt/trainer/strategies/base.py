from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
from chatgpt.replay_buffer import ReplayBuffer
from torch.utils.data import DataLoader


class Strategy(ABC):
    """
        Base class for training strategies.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setup_distributed()

    @abstractmethod
    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        pass

    @abstractmethod
    def optimizer_step(self, optimizer: optim.Optimizer, **kwargs) -> None:
        pass

    @abstractmethod
    def setup_distributed(self) -> None:
        pass

    @abstractmethod
    def setup_model(self, model: nn.Module) -> nn.Module:
        pass

    @abstractmethod
    def setup_optimizer(self, optimizer: optim.Optimizer, model: nn.Module) -> optim.Optimizer:
        pass

    @abstractmethod
    def setup_dataloader(self, replay_buffer: ReplayBuffer, pin_memory: bool = False) -> DataLoader:
        pass

    def model_init_context(self):
        return nullcontext()
