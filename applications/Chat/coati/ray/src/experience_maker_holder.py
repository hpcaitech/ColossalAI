import torch
from typing import Any, Callable, Dict, List, Optional, Union
import ray
from ray.exceptions import GetTimeoutError
from torch import Tensor
import torch.nn as nn
from coati.models.base import Actor, Critic, RewardModel
from coati.trainer.strategies.sampler import DistributedSampler
from coati.trainer.strategies import Strategy
from coati.experience_maker import NaiveExperienceMaker, Experience, ExperienceMaker

from copy import deepcopy
from threading import Lock
import time
import os
import tracemalloc

from .utils import is_rank_0, get_strategy_from_args, set_dist_env, get_actor_from_args, get_critic_from_args, \
    get_reward_model_from_args


@ray.remote(concurrency_groups={"experience_io": 1, "model_io": 1, "compute": 1})
class ExperienceMakerHolder:
    '''
    Args:
        detached_trainer_name_list: str list to get ray actor handles
        strategy: 
        experience_batch_size: batch size of generated experience
        kl_coef: the coefficient of kl divergence loss
    '''

    def __init__(self,
                 detached_trainer_name_list: List[str],
                 strategy: str,
                 env_info: Dict[str, str] = None,
                 experience_batch_size: int = 8,
                 kl_coef: float = 0.1,
                 **generate_kwargs):
        # set environment variables
        if env_info:
            set_dist_env(env_info=env_info)
        self.target_trainer_list = []
        for name in detached_trainer_name_list:
            self.target_trainer_list.append(ray.get_actor(name, namespace=os.environ["RAY_NAMESPACE"]))
        self.strategy_str = strategy
        self.strategy = get_strategy_from_args(strategy)
        self.experience_batch_size = experience_batch_size
        self.kl_coef = kl_coef
        self.generate_kwargs = generate_kwargs
        actor, critic, reward_model, initial_model = None, None, None, None
        self.experience_maker = NaiveExperienceMaker(actor, critic, reward_model, initial_model, self.kl_coef)

        self._model_visit_lock = Lock()
        self._initial_model_initialized = False
        self._reward_model_initialized = False
        self._actor_initialized = False
        self._critic_initialized = False

        if 'debug' in self.generate_kwargs and self.generate_kwargs['debug'] == True:
            self._debug = True
        else:
            self._debug = False
        self.target_auto_balance = False

        if self._debug:
            print('[maker] Waiting for INIT')

    def _get_ready(self):
        while not self._fully_initialized():
            time.sleep(1.0)

    def _fully_initialized(self):
        if not self._initial_model_initialized:
            return False
        if not self._reward_model_initialized:
            return False
        if not self._actor_initialized:
            return False
        if not self._critic_initialized:
            return False
        return True

    def update_target_trainer_list(self, detached_trainer_name_list):
        self.target_trainer_list = []
        for name in detached_trainer_name_list:
            self.target_trainer_list.append(ray.get_actor(name))

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
    def _send_experience(self, experience):
        if not self.target_auto_balance:
            # choose the trainer in polling mannar
            if not hasattr(self, "_target_idx"):
                self._target_idx = 0
            chosen_trainer = self.target_trainer_list[self._target_idx]
            if self._debug:
                print(f"[maker] sending exp to {chosen_trainer}")
            chosen_trainer.buffer_append.remote(experience)
            self._target_idx = (self._target_idx + 1) % len(self.target_trainer_list)
        else:
            # choose a trainer that has the least experience batch in its detached_replay_buffer
            chosen_trainer = None
            min_length = None
            if self._debug:
                print("[maker] choosing tartget trainer")
            while chosen_trainer is None:
                for target_trainer in self.target_trainer_list:
                    try:
                        temp_length = ray.get(target_trainer.buffer_get_length.remote(), timeout=0.1)
                        if min_length is None:
                            min_length = temp_length
                            chosen_trainer = target_trainer
                        else:
                            if temp_length < min_length:
                                min_length = temp_length
                                chosen_trainer = target_trainer
                    except GetTimeoutError:
                        pass
            if self._debug:
                print(f"[maker] sending exp to {chosen_trainer}")
            chosen_trainer.buffer_append.remote(experience)

    def workingloop(self, dataset, tokenizer: Optional[Callable[[Any], dict]] = None, times=5000 * 50000):
        self._get_ready()
        sampler = self.strategy.setup_sampler(dataset)
        for _ in range(times):
            rand_prompts = sampler.sample(self.experience_batch_size)
            if tokenizer is not None:
                inputs = tokenizer(rand_prompts)
            else:
                inputs = rand_prompts
            self._model_visit_lock.acquire()
            experience = self._make_experience(inputs=inputs)
            self._model_visit_lock.release()
            self._send_experience(experience=experience)

    @ray.method(concurrency_group="model_io")
    def initialize_experience_maker(self,
                                    actor_model: str = None,
                                    actor_pretrained: str = None,
                                    actor_state_dict: Dict[str, Any] = None,
                                    critic_model: str = None,
                                    critic_pretrained: str = None,
                                    critic_state_dict: Dict[str, Any] = None,
                                    chunk_start: bool = None,
                                    chunk_end: bool = None):
        '''
            called by trainer
            chunk_start: Set True at the first call. Before sending state_dict calls
            chunk_end: Set True at the last call. After sending state_dict calls.

            TODO: load_state_dict integrate with model-sharding strategy 
        '''
        if self._fully_initialized():
            return

        if chunk_start:
            if self._debug:
                print('[maker] INIT')
            with torch.no_grad():
                # (csric) any better way to get model structure?
                with self.strategy.model_init_context():
                    if not self._actor_initialized and actor_model is not None:
                        self.experience_maker.actor = get_actor_from_args(actor_model, actor_pretrained).half().requires_grad_(False)
                    if not self._critic_initialized and critic_model is not None:
                        self.experience_maker.critic = get_critic_from_args(critic_model, critic_pretrained).half().requires_grad_(False)
                    if not self._initial_model_initialized and actor_model is not None:
                        self.experience_maker.initial_model = get_actor_from_args(actor_model, actor_pretrained).half().requires_grad_(False)
                    if not self._reward_model_initialized and critic_model is not None:
                        self.experience_maker.reward_model = get_reward_model_from_args(critic_model, critic_pretrained).half().requires_grad_(False)

        with torch.no_grad():
            if not self._actor_initialized and actor_state_dict is not None:
                self.experience_maker.actor.model.load_state_dict(actor_state_dict, strict=False)
            if not self._critic_initialized and critic_state_dict is not None:
                self.experience_maker.critic.load_state_dict(critic_state_dict, strict=False)
            if not self._initial_model_initialized and actor_state_dict is not None:
                self.experience_maker.initial_model.model.load_state_dict(actor_state_dict, strict=False)
            if not self._reward_model_initialized and critic_state_dict is not None:
                self.experience_maker.reward_model.load_state_dict(critic_state_dict, strict=False)


        if chunk_end:
            with torch.no_grad():
                if actor_model is not None:
                    if not self._actor_initialized:
                        self.experience_maker.actor = self.strategy.prepare(self.experience_maker.actor.to(torch.cuda.current_device()))
                    if not self._initial_model_initialized:
                        self.experience_maker.initial_model = self.strategy.prepare(self.experience_maker.initial_model.to(torch.cuda.current_device()))
                    self._actor_initialized = True
                    self._initial_model_initialized = True
                if critic_model is not None:
                    if not self._critic_initialized:
                        self.experience_maker.critic = self.strategy.prepare(self.experience_maker.critic.to(torch.cuda.current_device()))
                    if not self._reward_model_initialized:
                        self.experience_maker.reward_model = self.strategy.prepare(self.experience_maker.reward_model.to(torch.cuda.current_device()))
                    self._critic_initialized = True
                    self._reward_model_initialized = True


    def initialize_experience_maker_local(self,
                                          initial_model_func=None,
                                          reward_model_func=None,
                                          actor_func=None,
                                          critic_func=None):
        '''
            Use function call to construct the model here, because some strategy requieres env_info
            The model initialized here will be IGNORED in initialize_experience_maker.
            initial_model and reward_model can have their own strategy rather than self.strategy. For example, Quantization.
        '''

        if actor_func is not None:
            self.experience_maker.actor = actor_func()
            self._actor_initialized = True
        if critic_func is not None:
            self.experience_maker.critic = critic_func()
            self._critic_initialized = True
        if initial_model_func is not None:
            self.experience_maker.initial_model = initial_model_func()
            self._initial_model_initialized = True
        if reward_model_func is not None:
            self.experience_maker.reward_model = reward_model_func()
            self._reward_model_initialized = True

    @ray.method(concurrency_group="model_io")
    def update_experience_maker(self,
                                new_actor_state_dict: Dict[str, Any] = None,
                                new_critic_state_dict: Dict[str, Any] = None,
                                chunk_start: bool = None,
                                chunk_end: bool = None):
        '''
            called by trainer
            chunk_start: Set True at the first call. Before sending state_dict calls
            chunk_end: Set True at the last call. After sending state_dict calls.
                
            TODO: load_state_dict integrate with model-sharding strategy
        '''
        _watch_memory = True
        if chunk_start:
            if self._debug:
                print("[maker] UPDATE ")
            if _watch_memory:
                tracemalloc.start()
            self._model_visit_lock.acquire()

        with torch.no_grad():
            if new_actor_state_dict is not None:
                self.experience_maker.actor.model.load_state_dict(new_actor_state_dict, strict=False)
            if new_critic_state_dict is not None:
                self.experience_maker.critic.load_state_dict(new_critic_state_dict, strict=False)

        if chunk_end:
            self._model_visit_lock.release()
            if _watch_memory:
                current, peak = tracemalloc.get_traced_memory()
                print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
                tracemalloc.stop()
