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


from .utils import is_rank_0, get_strategy_from_args, set_dist_env


@ray.remote(concurrency_groups={"experience_io": 1, "model_io": 1, "compute": 1})
class ExperienceMakerHolder:
    '''
    Args:
        detached_trainer_name_list: str list to get ray actor handleskkk
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
        # Need a trainer to give an actor and a critic via initialize_experience_maker(...)
        actor, critic, reward_model, initial_model = None, None, None, None
        self.experience_maker = NaiveExperienceMaker(actor, critic, reward_model, initial_model, self.kl_coef)
        self._model_visit_lock = Lock()
        self.fully_initialized = False
        if 'debug' in self.generate_kwargs and self.generate_kwargs['debug'] == True:
            print('[maker] Waiting for INIT')

    def _get_ready(self):
        while not self.fully_initialized:
            time.sleep(1.0)

    def update_target_trainer_list(self, detached_trainer_name_list):
        self.target_trainer_list = []
        for name in detached_trainer_name_list:
            self.target_trainer_list.append(ray.get_actor(name))

    # copy from ../trainer/base.py
    @ray.method(concurrency_group="compute")
    def _make_experience(self, inputs: Union[Tensor, Dict[str, Tensor]]) -> Experience:
        self._get_ready()
        if isinstance(inputs, Tensor):
            return self.experience_maker.make_experience(inputs, **self.generate_kwargs)
        elif isinstance(inputs, dict):
            return self.experience_maker.make_experience(**inputs, **self.generate_kwargs)
        else:
            raise ValueError(f'Unsupported input type "{type(inputs)}"')

    @ray.method(concurrency_group="experience_io")
    def _send_experience(self, experience):
        '''
        ignore it

        # choose a trainer that has the least experience batch in its detached_replay_buffer
        chosen_trainer = None
        min_length = None
        if 'debug' in self.generate_kwargs and self.generate_kwargs['debug'] == True:
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
                    
        if 'debug' in self.generate_kwargs and self.generate_kwargs['debug'] == True:
            print(f"[maker] sending exp to {chosen_trainer}")
        chosen_trainer.buffer_append.remote(experience)
        '''
        # 
        if not hasattr(self, "_target_idx"):
            self._target_idx = 0
        chosen_trainer = self.target_trainer_list[self._target_idx]
        if 'debug' in self.generate_kwargs and self.generate_kwargs['debug'] == True:
            print(f"[maker] sending exp to {chosen_trainer}")
        chosen_trainer.buffer_append.remote(experience)
        self._target_idx = (self._target_idx + 1) % len(self.target_trainer_list)

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
    def initialize_experience_maker(self, init_actor: Actor, init_critic: Critic):
        '''
        called by trainer. Only once.
        '''
        # TODO: reduce malloc
        if self.fully_initialized:
            return
        if 'debug' in self.generate_kwargs and self.generate_kwargs['debug'] == True:
            print('[maker] INIT')
        with torch.no_grad():
            with self.strategy.model_init_context():
                actor = init_actor
                critic = init_critic
                initial_model = deepcopy(actor)
                reward_model = RewardModel(deepcopy(critic.model),
                                           deepcopy(critic.value_head)).to(torch.cuda.current_device())
            if self.strategy_str != 'colossalai_gemini':
                actor.to(torch.float16).to(torch.cuda.current_device())
                critic.to(torch.float16).to(torch.cuda.current_device())
                initial_model.to(torch.float16).to(torch.cuda.current_device())
                reward_model.to(torch.float16).to(torch.cuda.current_device())

            self.experience_maker.actor = self.strategy.prepare(actor)
            self.experience_maker.critic = self.strategy.prepare(critic)
            self.experience_maker.initial_model = self.strategy.prepare(initial_model)
            self.experience_maker.reward_model = self.strategy.prepare(reward_model)
        self.fully_initialized = True

    @ray.method(concurrency_group="model_io")
    def update_experience_maker(self, new_actor: Actor, new_critic: Critic):
        '''
            called by trainer
        '''
        # TODO: reduce malloc
        self._model_visit_lock.acquire()
        with torch.no_grad():
            if 'debug' in self.generate_kwargs and self.generate_kwargs['debug'] == True:
                print("[maker] UPDATE ")
            if self.strategy_str != 'colossalai_gemini':
                new_actor.to(torch.float16).to(torch.cuda.current_device())
                new_critic.to(torch.float16).to(torch.cuda.current_device())
            self.experience_maker.actor = self.strategy.prepare(new_actor)
            self.experience_maker.critic = self.strategy.prepare(new_critic)
        self._model_visit_lock.release()
