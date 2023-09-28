from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List

import torch.nn as nn
import tqdm
from coati.experience_buffer import NaiveExperienceBuffer
from coati.experience_maker import Experience
from torch.optim import Optimizer

from .callbacks import Callback
from .strategies import Strategy
from .utils import is_rank_0


class SLTrainer(ABC):
    """
        Base class for supervised learning trainers.

    Args:
        strategy (Strategy):the strategy to use for training
        max_epochs (int, defaults to 1): the number of epochs of training process
        model (nn.Module): the model to train
        optim (Optimizer): the optimizer to use for training
    """

    def __init__(
        self,
        strategy: Strategy,
        max_epochs: int,
        model: nn.Module,
        optimizer: Optimizer,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.max_epochs = max_epochs
        self.model = model
        self.optimizer = optimizer

    @abstractmethod
    def _train(self, epoch):
        raise NotImplementedError()

    @abstractmethod
    def _eval(self, epoch):
        raise NotImplementedError()

    def _before_fit(self):
        raise NotImplementedError()

    def fit(self, *args, **kwargs):
        self._before_fit(*args, **kwargs)
        for epoch in tqdm.trange(self.max_epochs, desc="Epochs", disable=not is_rank_0()):
            self._train(epoch)
            self._eval(epoch)


class OnPolicyTrainer(ABC):
    """
        Base class for on-policy rl trainers, e.g. PPO.

    Args:
        strategy (Strategy):the strategy to use for training
        data_buffer (NaiveExperienceBuffer): the buffer to collect experiences
        sample_buffer (bool, defaults to False): whether to sample from buffer
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
    """

    def __init__(
        self,
        strategy: Strategy,
        data_buffer: NaiveExperienceBuffer,
        sample_buffer: bool,
        dataloader_pin_memory: bool,
        callbacks: List[Callback] = [],
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.data_buffer = data_buffer
        self.sample_buffer = sample_buffer
        self.dataloader_pin_memory = dataloader_pin_memory
        self.callbacks = callbacks

    @contextmanager
    def _fit_ctx(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_start()
        try:
            yield
        finally:
            for callback in self.callbacks:
                callback.on_fit_end()

    @contextmanager
    def _episode_ctx(self, episode: int) -> None:
        for callback in self.callbacks:
            callback.on_episode_start(episode)
        try:
            yield
        finally:
            for callback in self.callbacks:
                callback.on_episode_end(episode)

    def _on_make_experience_start(self) -> None:
        for callback in self.callbacks:
            callback.on_make_experience_start()

    def _on_make_experience_end(self, experience: Experience) -> None:
        for callback in self.callbacks:
            callback.on_make_experience_end(experience)

    def _on_learn_epoch_start(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_learn_epoch_start(epoch)

    def _on_learn_epoch_end(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_learn_epoch_end(epoch)

    def _on_learn_batch_start(self) -> None:
        for callback in self.callbacks:
            callback.on_learn_batch_start()

    def _on_learn_batch_end(self, experience: Experience) -> None:
        for callback in self.callbacks:
            callback.on_learn_batch_end(experience)

    @abstractmethod
    def _make_experience(self, collect_step: int):
        """
        Implement this method to make experience.
        """
        raise NotImplementedError()

    @abstractmethod
    def _learn(self, update_step: int):
        """
        Implement this method to learn from experience, either
        sample from buffer or transform buffer into dataloader.
        """
        raise NotImplementedError()

    def _collect_phase(self, collect_step: int):
        self._on_make_experience_start()
        experience = self._make_experience(collect_step)
        self._on_make_experience_end(experience)
        self.data_buffer.append(experience)

    def _update_phase(self, update_step: int):
        self._on_learn_epoch_start(update_step)
        self._learn(update_step)
        self._on_learn_epoch_end(update_step)

    def _before_fit(self, *args, **kwargs):
        raise NotImplementedError()

    def fit(
        self,
        num_episodes: int,
        num_collect_steps: int,
        num_update_steps: int,
        *args,
        **kwargs,
    ):
        """
        The main training loop of on-policy rl trainers.

        Args:
            num_episodes (int): the number of episodes to train
            num_collect_steps (int): the number of collect steps per episode
            num_update_steps (int): the number of update steps per episode
        """
        self._before_fit(*args, **kwargs)
        with self._fit_ctx():
            for episode in tqdm.trange(num_episodes, desc="Episodes", disable=not is_rank_0()):
                with self._episode_ctx(episode):
                    for collect_step in tqdm.trange(num_collect_steps, desc="Collect steps", disable=not is_rank_0()):
                        self._collect_phase(collect_step)
                    if not self.sample_buffer:
                        # HACK(cwher): according to the design of boost API, dataloader should also be boosted,
                        #  but it is impractical to adapt this pattern in RL training. Thus, I left dataloader unboosted.
                        #  I only call strategy.setup_dataloader() to setup dataloader.
                        self.dataloader = self.strategy.setup_dataloader(self.data_buffer, self.dataloader_pin_memory)
                    for update_step in tqdm.trange(num_update_steps, desc="Update steps", disable=not is_rank_0()):
                        self._update_phase(update_step)
                    # NOTE: this is for on-policy algorithms
                    self.data_buffer.clear()
