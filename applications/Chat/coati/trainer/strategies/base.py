from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from coati.replay_buffer import ReplayBuffer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .sampler import DistributedSampler

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


class Strategy(ABC):
    """
        Base class for training strategies.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setup_distributed()

    @abstractmethod
    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: Optimizer, **kwargs) -> None:
        pass

    @abstractmethod
    def optimizer_step(self, optimizer: Optimizer, **kwargs) -> None:
        pass

    @abstractmethod
    def setup_distributed(self) -> None:
        pass

    @abstractmethod
    def setup_model(self, model: nn.Module) -> nn.Module:
        pass

    @abstractmethod
    def setup_optimizer(self, optimizer: Optimizer, model: nn.Module) -> Optimizer:
        pass

    @abstractmethod
    def setup_dataloader(self, replay_buffer: ReplayBuffer, pin_memory: bool = False) -> DataLoader:
        pass

    def model_init_context(self):
        return nullcontext()

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        """Prepare models or model-optimizer-pairs based on each strategy.

        Example::
            >>> # when fine-tuning actor and critic
            >>> (actor, actor_optim), (critic, critic_optim), reward_model, initial_model = strategy.prepare((actor, actor_optim), (critic, critic_optim), reward_model, initial_model)
            >>> # or when training reward model
            >>> (reward_model, reward_model_optim) = strategy.prepare((reward_model, reward_model_optim))
            >>> # or just inference
            >>> actor, critic = strategy.prepare(actor, critic)

        Returns:
            Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]: Models or model-optimizer-pairs in the original order.
        """

        rets = []
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 2, f'Expect (model, optimizer) pair, got a tuple with size "{len(arg)}"'
                model, optimizer = arg
                model = self.setup_model(model)
                optimizer = self.setup_optimizer(optimizer, model)
                rets.append((model, optimizer))
            elif isinstance(arg, nn.Module):
                rets.append(self.setup_model(model))
            else:
                raise RuntimeError(f'Expect model or (model, optimizer) pair, got {type(arg)}')

        if len(rets) == 1:
            return rets[0]
        return rets

    @staticmethod
    def unwrap_model(model: nn.Module) -> nn.Module:
        """Get the unwrapped model from a wrapped model made by Strategy.prepare.

        Args:
            model (nn.Module): the model to unwrap

        Returns:
            nn.Module: the original model
        """
        return model

    @abstractmethod
    def save_model(self, model: nn.Module, path: str, only_rank0: bool = True) -> None:
        pass

    @abstractmethod
    def load_model(self, model: nn.Module, path: str, map_location: Any = None, strict: bool = True) -> None:
        pass

    @abstractmethod
    def save_optimizer(self, optimizer: Optimizer, path: str, only_rank0: bool = False) -> None:
        pass

    @abstractmethod
    def load_optimizer(self, optimizer: Optimizer, path: str, map_location: Any = None) -> None:
        pass

    def setup_sampler(self, dataset) -> DistributedSampler:
        return DistributedSampler(dataset, 1, 0)

    @abstractmethod
    def save_pretrained(self,
                        model: nn.Module,
                        path: str,
                        only_rank0: bool = True,
                        tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        pass

    @abstractmethod
    def get_model_state_dict_shard(self, model: nn.Module, **config):
        pass
