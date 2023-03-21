from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Optimizer
from chatgpt.experience_maker import Experience
from chatgpt.replay_buffer import DetachedReplayBuffer
from tqdm import tqdm

from .callbacks import Callback
from .strategies import Strategy
from .utils import is_rank_0

import ray


# @ray.remote
class DetachedTrainer(ABC):
    '''
        Base class for detached rlhf trainers. 
        'detach' means that the experience maker is detached compared to a normal Trainer.
        Please set name attribute during init:
            >>> trainer = DetachedTrainer.options(..., name = "xxx", ...).remote()
            So an ExperienceMakerHolder can reach the detached_replay_buffer by Actor's name.
    Args:
        detached_strategy (DetachedStrategy): the strategy to use for training
        detached_replay_buffer_ref (ObjectRef[DetachedReplayBuffer]): the replay buffer to use for training
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        data_loader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    '''

    def __init__(self,
                 experience_maker_holder_name_list: List[str],
                 strategy: Strategy,# TODO: DetachedStrategy
                 detached_replay_buffer: DetachedReplayBuffer = None,
                 experience_batch_size: int = 8,
                 max_epochs: int = 1,
                 dataloader_pin_memory: bool = True,
                 callbacks: List[Callback] = [],
                 **generate_kwargs
                 )->None:
        super().__init__()
        self.strategy = strategy
        self.detached_replay_buffer = detached_replay_buffer
        self.experience_batch_size = experience_batch_size
        self.max_epochs = max_epochs
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.callbacks = callbacks

        self.target_holder_name_list = experience_maker_holder_name_list
        self.target_holder_list = []

    def update_target_holder_list(self, experience_maker_holder_name_list):
        self.target_holder_name_list = experience_maker_holder_name_list
        self.target_holder_list = []
        for name in self.target_holder_name_list:
            self.target_holder_list.append(ray.get_actor(name))

    @abstractmethod
    def update_remote_makers(self):
        pass

    @abstractmethod
    def training_step(self, experience: Experience) -> Dict[str, Any]:
        pass

    def _learn(self):
        pbar = tqdm(range(self.max_epochs), desc='Train epoch', disable=not is_rank_0())
        for _ in pbar:
            experience = self.detached_replay_buffer.sample()
            metrics = self.training_step(experience)
            pbar.set_postfix(metrics)

    def fit(self, num_episodes: int = 50000, max_timesteps: int = 500, update_timesteps: int = 5000) -> None:
        self._on_fit_start()
        for episode in range(num_episodes):
            self._on_episode_start(episode)
            for timestep in tqdm(range(max_timesteps),
                                 desc=f'Episode [{episode+1}/{num_episodes}]',
                                 disable=not is_rank_0()):
                for _ in update_timesteps:
                    self._learn()
                # assume those remote holders are working
                # self.update_remote_makers()

            self._on_episode_end(episode)
        self._on_fit_end()

    def buffer_get_length(self):
        # called by ExperienceMakerHolder
        return self.detached_replay_buffer.get_length()

    def buffer_append(self, experience: Experience):
        # called by ExperienceMakerHolder
        self.detached_replay_buffer.append(experience)

    def strategy_save_model(self, model: nn.Module, path: str, only_rank0: bool = False) -> None:
        self.strategy.save_model(model, path, only_rank0)

    def strategy_save_potimizer(self, optimizer: Optimizer, path: str, only_rank0: bool = False) -> None:
        self.strategy.save_optimizer(optimizer, path, only_rank0)

    def _on_fit_start(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_start()

    def _on_fit_end(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_end()

    def _on_episode_start(self, episode: int) -> None:
        for callback in self.callbacks:
            callback.on_episode_start(episode)

    def _on_episode_end(self, episode: int) -> None:
        for callback in self.callbacks:
            callback.on_episode_end(episode)