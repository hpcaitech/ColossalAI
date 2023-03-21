import torch
from typing import Any, Callable, Dict, List, Optional, Union
from .naive import NaiveExperienceMaker, Experience, ExperienceMaker
from ..replay_buffer.detached import DetachedReplayBuffer
import ray
from torch import Tensor
import torch.nn as nn
from chatgpt.models.base import Actor
from chatgpt.trainer.strategies.sampler import DistributedSampler

@ray.remote
class ExperienceMakerHolder:
    '''
    Args:
        detached_trainer_name_list: str list to get ray actor handles
        actor: \
        critic: \
        reward_model: \
        initial_model: \
        kl_coef:                 NaiveExperienceMaker init
        experience_batch_size: batch size of generated experience
    '''

    def __init__(self, 
                 detached_trainer_name_list: List[str], 
                 actor: Actor,
                 critic: nn.Module,
                 reward_model: nn.Module,
                 initial_model: Actor,
                 kl_coef: float = 0.1,
                 experience_batch_size:int = 8,
                 **generate_kwargs):
        
        self.experience_maker = NaiveExperienceMaker(actor, critic, reward_model, initial_model,kl_coef)
        self.target_trainer_list = []
        self.experience_batch_size = experience_batch_size
        self.generate_kwargs = generate_kwargs
        for name in detached_trainer_name_list:
            self.target_trainer_list.append(ray.get_actor(name))
    
    # copy from ../trainer/base.py
    def _make_experience(self, inputs: Union[Tensor, Dict[str, Tensor]]) -> Experience:
        if isinstance(inputs, Tensor):
            return self.experience_maker.make_experience(inputs, **self.generate_kwargs)
        elif isinstance(inputs, dict):
            return self.experience_maker.make_experience(**inputs, **self.generate_kwargs)
        else:
            raise ValueError(f'Unsupported input type "{type(inputs)}"')
        
    def update_target_trainer_list(self, detached_trainer_name_list):
        self.target_trainer_list = []
        for name in detached_trainer_name_list:
            self.target_trainer_list.append(ray.get_actor(name))

    def make_and_send(self, inputs):
        experience = self._make_experience(inputs)
        # choose a trainer that has the least experience batch in its detached_replay_buffer
        chosen_trainer = None
        min_length = None
        if 'debug' in self.generate_kwargs and self.generate_kwargs['debug'] == True:
            print("[maker] choosing tartget")
        while chosen_trainer is None:
            for target_trainer in self.target_trainer_list:
                temp_length = ray.get(target_trainer.buffer_get_length.remote())
                if min_length is None:
                    min_length = temp_length
                    chosen_trainer = target_trainer
                else:
                    if temp_length < min_length:
                        min_length = temp_length
                        chosen_trainer = target_trainer
        if 'debug' in self.generate_kwargs and self.generate_kwargs['debug'] == True:
            print("[maker] sending")
        chosen_trainer.buffer_append.remote(experience)

    def workingloop(self, sampler: DistributedSampler, tokenizer: Optional[Callable[[Any], dict]] = None, times=5000 * 50000):
        for _ in range(times):
            rand_prompts = sampler.sample(self.experience_batch_size)
            if tokenizer is not None:
                    inputs = tokenizer(rand_prompts)
            else:
                inputs = rand_prompts
            self.make_and_send(inputs)
        
    def update_experience_maker(self, new_actor, new_critic):
        # TODO: parameter update
        '''
        pseudo:
            self.experience_maker.actor.update()
            self.experience_maker.critic.update()
        '''
        # TODO: reduce malloc
        with torch.no_grad():
            self.experience_maker.actor = new_actor
            self.experience_maker.critic = new_critic
        pass
        