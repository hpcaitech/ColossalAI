from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from coati.replay_buffer import ReplayBuffer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from colossalai.booster import Booster
from colossalai.booster.plugin import Plugin

from .sampler import DistributedSampler

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


class Strategy(ABC):
    """
        Base class for training strategies.
    """

    def __init__(self, plugin_initializer: Callable[..., Optional[Plugin]] = lambda: None) -> None:
        super().__init__()
        if not dist.is_initialized():
            # NOTE: dist must be initialized before Booster
            self.setup_distributed()
        self.plugin = plugin_initializer()
        self.booster = Booster(plugin=self.plugin)
        self._post_init()

    @abstractmethod
    def _post_init(self) -> None:
        pass

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: Optimizer, **kwargs) -> None:
        self.booster.backward(loss, optimizer)

    def optimizer_step(self, optimizer: Optimizer, **kwargs) -> None:
        optimizer.step()

    @abstractmethod
    def setup_distributed(self) -> None:
        pass

    def setup_model(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError()

    def setup_optimizer(self, optimizer: Optimizer, model: nn.Module) -> Optimizer:
        raise NotImplementedError()

    def setup_model_optimizer(self,
                              model: nn.Module,
                              optimizer: Optimizer
                              ) -> Tuple[nn.Module, Optimizer]:
        # NOTE: Booster.boost must take in both model and optimizer
        model, optimizer, *_ = self.booster.boost(model=model,
                                                  optimizer=optimizer)
        return model, optimizer

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
                model, optimizer = self.setup_model_optimizer(*arg)
                rets.append((model, optimizer))
            elif isinstance(arg, nn.Module):
                rets.append(self.setup_model(arg))
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

    def save_model(self,
                   model: nn.Module,
                   path: str,
                   only_rank0: bool = True,
                   **kwargs
                   ) -> None:
        self.booster.save_model(model, path, shard=not only_rank0, **kwargs)

    def load_model(self, model: nn.Module, path: str, strict: bool = True) -> None:
        self.booster.load_model(model, path, strict)

    def save_optimizer(self,
                       optimizer: Optimizer,
                       path: str,
                       only_rank0: bool = False,
                       **kwargs
                       ) -> None:
        self.booster.save_optimizer(optimizer, path, shard=not only_rank0, **kwargs)

    def load_optimizer(self, optimizer: Optimizer, path: str) -> None:
        self.booster.load_optimizer(optimizer, path)

    def setup_sampler(self, dataset) -> DistributedSampler:
        # FIXME(cwher): this is only invoked in train_on_ray, not tested after adapt Boost API.
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
