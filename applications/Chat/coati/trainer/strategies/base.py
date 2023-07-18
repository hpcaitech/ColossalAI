from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from coati.replay_buffer import ReplayBuffer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from colossalai.booster import Booster
from colossalai.booster.plugin import Plugin

from .sampler import DistributedSampler

_BoostArgSpec = Union[nn.Module, Tuple[nn.Module, Optimizer], Dict]


class Strategy(ABC):
    """
        Base class for training strategies.
    """

    def __init__(self, plugin_initializer: Callable[..., Optional[Plugin]] = lambda: None) -> None:
        super().__init__()
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

    @abstractmethod
    def setup_dataloader(self, replay_buffer: ReplayBuffer, pin_memory: bool = False) -> DataLoader:
        pass

    def model_init_context(self):
        return nullcontext()

    def prepare(self, *boost_args: _BoostArgSpec) -> Union[List[_BoostArgSpec], _BoostArgSpec]:
        """Prepare [model | (model, optimizer) | Dict] based on each strategy.
        NOTE: the keys of Dict must be a subset of `self.booster.boost`'s arguments.

        Example::
            >>> # e.g., include lr_scheduler
            >>> result_dict = strategy.prepare(dict(model=model, lr_scheduler=lr_scheduler))
            >>> # when fine-tuning actor and critic
            >>> (actor, actor_optim), (critic, critic_optim), reward_model, initial_model = strategy.prepare((actor, actor_optim), (critic, critic_optim), reward_model, initial_model)
            >>> # or when training reward model
            >>> (reward_model, reward_model_optim) = strategy.prepare((reward_model, reward_model_optim))
            >>> # or just inference
            >>> actor, critic = strategy.prepare(actor, critic)

        Returns:
            Union[List[_BoostArgSpec], _BoostArgSpec]: [model | (model, optimizer) | Dict] in the original order.
        """

        rets = []
        for arg in boost_args:
            if isinstance(arg, nn.Module):
                model, *_ = self.booster.boost(arg)
                rets.append(model)
            elif isinstance(arg, tuple):
                try:
                    model, optimizer = arg
                except ValueError:
                    raise RuntimeError(f'Expect (model, optimizer) pair, got a tuple with size "{len(arg)}"')
                model, optimizer, *_ = self.booster.boost(model=model,
                                                          optimizer=optimizer)
                rets.append((model, optimizer))
            elif isinstance(arg, Dict):
                model, optimizer, criterion, dataloader, lr_scheduler = self.booster.boost(**arg)
                boost_result = dict(model=model,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    dataloader=dataloader,
                                    lr_scheduler=lr_scheduler)
                # remove None values
                boost_result = {
                    key: value
                    for key, value in boost_result.items() if value is not None
                }
                rets.append(boost_result)
            else:
                raise RuntimeError(f'Type {type(arg)} is not supported')

        return rets[0] if len(rets) == 1 else rets

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
