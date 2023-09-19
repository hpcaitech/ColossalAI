import os
import time
import tracemalloc
from threading import Lock
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import ray
import torch
from coati.experience_buffer.utils import split_experience_batch
from coati.experience_maker import Experience, NaiveExperienceMaker
from coati.models.base import Actor, Critic, RewardModel
from coati.trainer.strategies import Strategy
from torch import Tensor
from tqdm import tqdm

from .callbacks import ExperienceMakerPerformanceEvaluator, MakerCallback
from .lora_constructor import LoRAConstructor
from .utils import get_model_numel, get_rank, is_rank_0, set_dist_env, state_dict_to


@ray.remote(concurrency_groups={"experience_io": 1, "model_io": 1, "compute": 1})
class ExperienceMakerHolder:
    """
    Args:
        detached_trainer_name_list: str list to get ray actor handles
        strategy:
        kl_coef: the coefficient of kl divergence loss
        sync_models_from_trainers: whether to sync models from trainers. If True, you must call sync_models_to_remote_makers() in trainers to sync models.
    """

    def __init__(
        self,
        detached_trainer_name_list: List[str],
        strategy_fn: Callable[[], Strategy],
        # a function returns (actor, critic, reward_model, initial_model)
        model_fn: Callable[[], Tuple[Actor, Critic, RewardModel, Actor]],
        env_info: Dict[str, str] = None,
        sync_models_from_trainers: bool = False,
        buffer_cpu_offload: bool = True,
        kl_coef: float = 0.1,
        callbacks: List[MakerCallback] = [],
        eval_performance: bool = False,
        debug: bool = False,
        update_lora_weights: bool = False,
        **generate_kwargs,
    ):
        # set environment variables
        if env_info:
            set_dist_env(env_info=env_info)
        self.target_trainer_list = []
        assert len(detached_trainer_name_list) > 0
        self._detached_trainer_name_list = detached_trainer_name_list
        self.strategy = strategy_fn()
        self.buffer_cpu_offload = buffer_cpu_offload
        self.kl_coef = kl_coef
        # init models
        with self.strategy.model_init_context():
            actor, critic, reward_model, initial_model = model_fn()
        self.generate_kwargs = _set_default_generate_kwargs(generate_kwargs, actor)
        if eval_performance:
            actor_numel = get_model_numel(actor)
            critic_numel = get_model_numel(critic)
            initial_model_numel = get_model_numel(initial_model)
            reward_model_numel = get_model_numel(reward_model)
            evaluator = ExperienceMakerPerformanceEvaluator(
                actor_numel, critic_numel, initial_model_numel, reward_model_numel
            )
            callbacks = callbacks + [evaluator]

        actor, critic, reward_model, initial_model = self.strategy.prepare(actor, critic, reward_model, initial_model)
        self.experience_maker = NaiveExperienceMaker(actor, critic, reward_model, initial_model, self.kl_coef)
        self.callbacks = callbacks

        self._model_visit_lock = Lock()

        self._is_fully_initialized = not sync_models_from_trainers

        self._debug = debug
        self._update_lora_weights = update_lora_weights
        if self._update_lora_weights:
            self.actor_lora_constructor = LoRAConstructor()
            self.critic_lora_constructor = LoRAConstructor()

        self.target_auto_balance = False

        self._target_idx = 0

        if self._debug:
            print(f"[maker{get_rank()}] will send items to {self._detached_trainer_name_list}")
            if not self._is_fully_initialized:
                print(f"[maker{get_rank()}] Waiting for INIT")

    def _get_ready(self):
        while not self._fully_initialized():
            time.sleep(1.0)

    def _fully_initialized(self):
        return self._is_fully_initialized

    def _init_target_trainer_list(self):
        if len(self.target_trainer_list) > 0:
            return
        for name in self._detached_trainer_name_list:
            self.target_trainer_list.append(ray.get_actor(name, namespace=os.environ["RAY_NAMESPACE"]))

    # copy from ../trainer/base.py
    @ray.method(concurrency_group="compute")
    def _make_experience(self, inputs: Union[Tensor, Dict[str, Tensor]]) -> Experience:
        if isinstance(inputs, Tensor):
            return self.experience_maker.make_experience(inputs, **self.generate_kwargs)
        elif isinstance(inputs, dict):
            return self.experience_maker.make_experience(**inputs, **self.generate_kwargs)
        else:
            raise ValueError(f'Unsupported input type "{type(inputs)}"')

    @ray.method(concurrency_group="experience_io")
    def _send_items(self, experience: Experience) -> None:
        self._init_target_trainer_list()
        items = split_experience_batch(experience)
        items_per_trainer = [[] for _ in range(len(self.target_trainer_list))]
        for item in items:
            items_per_trainer[self._target_idx].append(item)
            self._target_idx = (self._target_idx + 1) % len(self.target_trainer_list)
        for i, target_trainer in enumerate(self.target_trainer_list):
            if len(items_per_trainer[i]) > 0:
                target_trainer.buffer_extend.remote(items_per_trainer[i])

    def _inference_step(self, batch) -> None:
        self._on_batch_start()
        with self._model_visit_lock:
            self._on_make_experience_start()
            experience = self._make_experience(batch)
            self._on_make_experience_end(experience)
        self._on_send_start()
        if self.buffer_cpu_offload:
            experience.to_device("cpu")
        self._send_items(experience)
        self._on_send_end()
        self._on_batch_end()

    def workingloop(self, dataloader_fn: Callable[[], Iterable], num_epochs: int = 1, num_steps: int = 0):
        """Working loop of the experience maker.

        Args:
            dataloader_fn (Callable[[], Iterable]): A function that returns a dataloader.
            num_epochs (int, optional): Iterate the dataloader for number of epochs. Defaults to 1.
            num_steps (int, optional): Iterate the dataloader for number if steps. If this value > 0, num_epochs will be ignored. Defaults to 0.
        """
        self._get_ready()
        self._on_loop_start()
        dataloader = dataloader_fn()
        if num_steps > 0:
            # ignore num epochs
            it = iter(dataloader)
            for _ in tqdm(range(num_steps), desc="ExperienceMaker", disable=not is_rank_0()):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(dataloader)
                    batch = next(it)
                self._inference_step(batch)
        else:
            with tqdm(total=num_epochs * len(dataloader), desc="ExperienceMaker", disable=not is_rank_0()) as pbar:
                for _ in range(num_epochs):
                    for batch in dataloader:
                        self._inference_step(batch)
                        pbar.update()
        self._on_loop_end()

    @ray.method(concurrency_group="model_io")
    def update_experience_maker(
        self,
        new_actor_state_dict: Dict[str, Any] = None,
        new_actor_lora_config_dict: Dict[str, Any] = None,
        new_critic_state_dict: Dict[str, Any] = None,
        new_critic_lora_config_dict: Dict[str, Any] = None,
        fully_update: bool = False,
        chunk_start: bool = None,
        chunk_end: bool = None,
    ):
        """
        called by trainer
        chunk_start: Set True at the first call. Before sending state_dict calls
        chunk_end: Set True at the last call. After sending state_dict calls.
        fully_update: Set True if you want to sync models when initializing

        TODO: load_state_dict integrate with model-sharding strategy
        """
        _watch_memory = self._debug
        if chunk_start:
            if self._debug:
                print("[maker] UPDATE ")
            if _watch_memory:
                tracemalloc.start()
            self._model_visit_lock.acquire()

        with torch.no_grad():
            if new_actor_state_dict is not None:
                if not self._update_lora_weights or fully_update:
                    self.experience_maker.actor.model.load_state_dict(new_actor_state_dict, strict=False)
                else:
                    new_actor_state_dict = state_dict_to(new_actor_state_dict, device=torch.cuda.current_device())
                    state_dict_increase = self.actor_lora_constructor.reconstruct_increase(
                        new_actor_state_dict, new_actor_lora_config_dict
                    )
                    self.actor_lora_constructor.load_state_dict_increase(
                        self.experience_maker.actor.model, state_dict_increase
                    )
            if new_critic_state_dict is not None:
                if not self._update_lora_weights or fully_update:
                    self.experience_maker.critic.load_state_dict(new_critic_state_dict, strict=False)
                else:
                    new_critic_state_dict = state_dict_to(new_critic_state_dict, device=torch.cuda.current_device())
                    state_dict_increase = self.critic_lora_constructor.reconstruct_increase(
                        new_critic_state_dict, new_critic_lora_config_dict
                    )
                    self.critic_lora_constructor.load_state_dict_increase(
                        self.experience_maker.critic, state_dict_increase
                    )

        # the lock must be released after both actor and critic being updated
        if chunk_end:
            self._model_visit_lock.release()
            if _watch_memory:
                current, peak = tracemalloc.get_traced_memory()
                print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
                tracemalloc.stop()
            if fully_update:
                self._is_fully_initialized = True

    def _on_make_experience_start(self) -> None:
        for callback in self.callbacks:
            callback.on_make_experience_start()

    def _on_make_experience_end(self, experience: Experience) -> None:
        for callback in self.callbacks:
            callback.on_make_experience_end(experience)

    def _on_loop_start(self) -> None:
        for callback in self.callbacks:
            callback.on_loop_start()

    def _on_loop_end(self) -> None:
        for callback in self.callbacks:
            callback.on_loop_end()

    def _on_send_start(self) -> None:
        for callback in self.callbacks:
            callback.on_send_start()

    def _on_send_end(self) -> None:
        for callback in self.callbacks:
            callback.on_send_end()

    def _on_batch_start(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_start()

    def _on_batch_end(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_end()


def _set_default_generate_kwargs(generate_kwargs: dict, actor: Actor) -> None:
    origin_model = actor.model
    new_kwargs = {**generate_kwargs}
    # use huggingface models method directly
    if "prepare_inputs_fn" not in generate_kwargs and hasattr(origin_model, "prepare_inputs_for_generation"):
        new_kwargs["prepare_inputs_fn"] = origin_model.prepare_inputs_for_generation

    if "update_model_kwargs_fn" not in generate_kwargs and hasattr(origin_model, "_update_model_kwargs_for_generation"):
        new_kwargs["update_model_kwargs_fn"] = origin_model._update_model_kwargs_for_generation

    return new_kwargs
