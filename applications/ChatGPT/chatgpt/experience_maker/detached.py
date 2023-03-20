import torch
from typing import Any, Callable, Dict, List, Optional, Union
from .naive import NaiveExperienceMaker, Experience, ExperienceMaker
from ..replay_buffer.detached import DetachedReplayBuffer
import ray
from torch import Tensor

class ExperienceMakerHolder:
    '''
    Args:
        detached_trainer_name_list: str list to get ray actor handles
        experience_maker: experience maker
    '''

    def __init__(self, detached_trainer_name_list: List[str], experience_maker : ExperienceMaker):
        self.experience_maker = experience_maker
        self.target_trainer_list = []
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
        while chosen_trainer is None:
            for target_trainer in self.target_trainer_list:
                temp_length = ray.get(target_trainer.get_buffer_length.remote())
                if min_length is None:
                    min_length = temp_length
                    chosen_trainer = target_trainer
                else:
                    if temp_length < min_length:
                        min_length = temp_length
                        chosen_trainer = target_trainer
        chosen_trainer.buffer_append.remote(experience)

    def update_experience_maker(self):
        # TODO: parameter update
        '''
        self.experience_maker.actor.update()
        self.experience_maker.critic.update()
        '''
        pass