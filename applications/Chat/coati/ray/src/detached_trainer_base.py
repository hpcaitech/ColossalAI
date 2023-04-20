from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from tqdm import tqdm
from coati.trainer.callbacks import Callback
from coati.experience_maker import Experience
import ray
import os

from .detached_replay_buffer import DetachedReplayBuffer
from .utils import is_rank_0

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
                 train_batch_size: int = 8,
                 buffer_limit: int = 0,
                 buffer_cpu_offload: bool = True,
                 experience_batch_size: int = 8,
                 max_epochs: int = 1,
                 dataloader_pin_memory: bool = True,
                 callbacks: List[Callback] = [],
                 **generate_kwargs) -> None:
        super().__init__()
        self.detached_replay_buffer = DetachedReplayBuffer(train_batch_size, limit=buffer_limit, cpu_offload=buffer_cpu_offload)
        self.experience_batch_size = experience_batch_size
        self.max_epochs = max_epochs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.callbacks = callbacks
        self.generate_kwargs = generate_kwargs
        self.target_holder_name_list = experience_maker_holder_name_list
        self.target_holder_list = []

    def update_target_holder_list(self, experience_maker_holder_name_list):
        self.target_holder_name_list = experience_maker_holder_name_list
        self.target_holder_list = []
        for name in self.target_holder_name_list:
            self.target_holder_list.append(ray.get_actor(name, namespace=os.environ["RAY_NAMESPACE"]))

    @abstractmethod
    def _update_remote_makers(self):
        pass

    @abstractmethod
    def training_step(self, experience: Experience) -> Dict[str, Any]:
        pass

    def _learn(self):
        pbar = tqdm(range(self.max_epochs), desc='Train epoch', disable=not is_rank_0())
        for _ in pbar:
            if 'debug' in self.generate_kwargs and self.generate_kwargs['debug'] == True:
                print("[trainer] sampling exp")
            experience = self._buffer_sample()
            if 'debug' in self.generate_kwargs and self.generate_kwargs['debug'] == True:
                print("[trainer] training step")
            metrics = self.training_step(experience)
            if 'debug' in self.generate_kwargs and self.generate_kwargs['debug'] == True:
                print("[trainer] step over")
            pbar.set_postfix(metrics)

    def fit(self, num_episodes: int = 50000, max_timesteps: int = 500, update_timesteps: int = 5000) -> None:
        self._on_fit_start()
        for episode in range(num_episodes):
            self._on_episode_start(episode)
            for timestep in tqdm(range(max_timesteps // update_timesteps),
                                 desc=f'Episode [{episode+1}/{num_episodes}]',
                                 disable=not is_rank_0()):
                self._learn()
                self._update_remote_makers()
            self._on_episode_end(episode)
        self._on_fit_end()

    @ray.method(concurrency_group="buffer_length")
    def buffer_get_length(self):
        # called by ExperienceMakerHolder
        if 'debug' in self.generate_kwargs and self.generate_kwargs['debug'] == True:
            print("[trainer]                telling length")
        return self.detached_replay_buffer.get_length()

    @ray.method(concurrency_group="buffer_append")
    def buffer_append(self, experience: Experience):
        # called by ExperienceMakerHolder
        if 'debug' in self.generate_kwargs and self.generate_kwargs['debug'] == True:
            # print(f"[trainer] receiving exp. Current buffer length: {self.detached_replay_buffer.get_length()}")
            print(f"[trainer]               receiving exp.")
        self.detached_replay_buffer.append(experience)

    @ray.method(concurrency_group="buffer_sample")
    def _buffer_sample(self):
        return self.detached_replay_buffer.sample()

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
