import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import ray
import torch
from coati.experience_buffer.utils import BufferItem
from coati.experience_maker import Experience
from torch.utils.data import DataLoader
from tqdm import tqdm

from .callbacks import TrainerCallback
from .detached_replay_buffer import DetachedReplayBuffer
from .utils import is_rank_0


class DetachedTrainer(ABC):
    """
        Base class for detached rlhf trainers.
        'detach' means that the experience maker is detached compared to a normal Trainer.
        Please set name attribute during init:
            >>> trainer = DetachedTrainer.options(..., name = "xxx", ...).remote()
            So an ExperienceMakerHolder can reach the detached_replay_buffer by Actor's name.
    Args:
        detached_strategy (DetachedStrategy): the strategy to use for training
        detached_replay_buffer_ref (ObjectRef[DetachedReplayBuffer]): the replay buffer to use for training
        data_loader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating

    """

    def __init__(
        self,
        experience_maker_holder_name_list: List[str],
        train_batch_size: int = 8,
        buffer_limit: int = 0,
        dataloader_pin_memory: bool = True,
        callbacks: List[TrainerCallback] = [],
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.detached_replay_buffer = DetachedReplayBuffer(train_batch_size, limit=buffer_limit)
        self.dataloader_pin_memory = dataloader_pin_memory
        self.callbacks = callbacks
        self.target_holder_name_list = experience_maker_holder_name_list
        self.target_holder_list = []
        self._is_target_holder_initialized = False
        self._debug = debug

    def update_target_holder_list(self):
        # as the length of target_holder_list may be zero, we need to check it by a bool flag
        if not self._is_target_holder_initialized:
            for name in self.target_holder_name_list:
                self.target_holder_list.append(ray.get_actor(name, namespace=os.environ["RAY_NAMESPACE"]))
            self._is_target_holder_initialized = True

    @abstractmethod
    def _update_remote_makers(self, fully_update: bool = False, **kwargs):
        pass

    def sync_models_to_remote_makers(self, **kwargs):
        self._update_remote_makers(fully_update=True, **kwargs)

    @abstractmethod
    def training_step(self, experience: Experience) -> Dict[str, Any]:
        pass

    def _learn(self, update_steps: int, train_epochs: int) -> None:
        data = []
        # warmup
        pbar = tqdm(range(update_steps), desc=f"Train epoch [1/{train_epochs}]", disable=not is_rank_0())
        self._on_epoch_start(0)
        self._learn_epoch(pbar, data)
        self._on_epoch_end(0)
        # item is already a batch
        dataloader = DataLoader(
            data, batch_size=1, shuffle=True, pin_memory=self.dataloader_pin_memory, collate_fn=lambda x: x[0]
        )
        for epoch in range(1, train_epochs):
            pbar = tqdm(dataloader, desc=f"Train epoch [{epoch + 1}/{train_epochs}]", disable=not is_rank_0())
            self._on_epoch_start(epoch)
            self._learn_epoch(pbar, data)
            self._on_epoch_end(epoch)

    def _learn_epoch(self, pbar: tqdm, data: List[Experience]) -> None:
        is_warmup = len(data) == 0
        for x in pbar:
            if self._debug:
                print("[trainer] training step")
            # sample a batch and then train to avoid waiting
            experience = x if not is_warmup else self._buffer_sample()
            experience.to_device(torch.cuda.current_device())
            self._on_batch_start()
            metrics = self.training_step(experience)
            self._on_batch_end(metrics, experience)

            if self._debug:
                print("[trainer] step over")
            experience.to_device("cpu")
            if is_warmup:
                data.append(experience)
            pbar.set_postfix(metrics)

    def fit(self, total_steps: int, update_steps: int, train_epochs: int = 1) -> None:
        self._on_fit_start()
        for i in tqdm(range(total_steps // update_steps), desc="Trainer", disable=not is_rank_0()):
            self._on_episode_start(i)
            self._learn(update_steps, train_epochs)
            self._on_update_start()
            self._update_remote_makers()
            self._on_update_end()
            self._on_episode_end(i)
        self._on_fit_end()

    @ray.method(concurrency_group="buffer_length")
    def buffer_get_length(self):
        # called by ExperienceMakerHolder
        if self._debug:
            print("[trainer]                telling length")
        return self.detached_replay_buffer.get_length()

    @ray.method(concurrency_group="buffer_append")
    def buffer_append(self, experience: Experience):
        # called by ExperienceMakerHolder
        if self._debug:
            print(f"[trainer]               receiving exp.")
        self.detached_replay_buffer.append(experience)

    @ray.method(concurrency_group="buffer_append")
    def buffer_extend(self, items: List[BufferItem]):
        # called by ExperienceMakerHolder
        if self._debug:
            print(f"[trainer]               receiving exp.")
        self.detached_replay_buffer.extend(items)

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

    def _on_epoch_start(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_epoch_start(epoch)

    def _on_epoch_end(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def _on_batch_start(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_start()

    def _on_batch_end(self, metrics: dict, experience: Experience) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(metrics, experience)

    def _on_update_start(self) -> None:
        for callback in self.callbacks:
            callback.on_update_start()

    def _on_update_end(self) -> None:
        for callback in self.callbacks:
            callback.on_update_end()
