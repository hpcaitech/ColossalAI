from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from coati.replay_buffer import ReplayBuffer
from coati.models.base import LM, RewardModel
from coati.models.lora import LoraLinear
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

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

    def save_model(self, model: nn.Module, path: str, only_rank0: bool = False, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        for module in model.modules():
            if isinstance(module, LoraLinear):
                module.merge_weights = True
                module.eval()
        
        if isinstance(model, RewardModel):
            state_dict = model.state_dict()
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
                torch.save(state_dict, path)

    def load_model(self, model: nn.Module, path: str, map_location: Any = None, strict: bool = True) -> None:
        unwrapped_model = self._unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    def save_optimizer(self, optimizer: Optimizer, path: str, only_rank0: bool = False) -> None:
        torch.save(optimizer.state_dict(), path)

    def load_optimizer(self, optimizer: Optimizer, path: str, map_location: Any = None) -> None:
        state_dict = torch.load(path, map_location=map_location)
        optimizer.load_state_dict(state_dict)
