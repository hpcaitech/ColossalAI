import torch
from typing import Any, Callable, Dict, List, Optional, Union
from .naive import NaiveExperienceMaker, Experience, ExperienceMaker
from ..replay_buffer.distributed import DistReplayBuffer
import ray
from torch import Tensor

class ExperienceMakerHolder:
    '''
    Args:
        dist_replay_buffer_name_list: str list to get ray actor handles
        experience_maker: experience maker
    '''

    def __init__(self, dist_replay_buffer_name_list: List[str], experience_maker : NaiveExperienceMaker):
        self.experience_maker = experience_maker
        self.target_buffer_list = []
        for name in dist_replay_buffer_name_list:
            self.target_buffer_list.append(ray.get_actor(name))
    
    # copy from ../trainer/base.py
    def _make_experience(self, inputs: Union[Tensor, Dict[str, Tensor]]) -> Experience:
        if isinstance(inputs, Tensor):
            return self.experience_maker.make_experience(inputs, **self.generate_kwargs)
        elif isinstance(inputs, dict):
            return self.experience_maker.make_experience(**inputs, **self.generate_kwargs)
        else:
            raise ValueError(f'Unsupported input type "{type(inputs)}"')
        
    def update_target_buffer_list(self, new_target_buffer_list):
        self.target_buffer_list = new_target_buffer_list

    def make_and_send(self, inputs):
        experience = self._make_experience(inputs)
        # choose a buffer that has the least experience batch
        chosen_buffer = None
        min_length = None
        while chosen_buffer is None:
            for target_buffer in self.target_buffer_list:
                temp_length = ray.get(target_buffer.get_length.remote())
                if min_length is None:
                    min_length = temp_length
                    chosen_buffer = target_buffer
                else:
                    if temp_length < min_length:
                        min_length = temp_length
                        chosen_buffer = target_buffer
        target_buffer.append.remote(experience)

    def update_experience_maker(self):
        # TODO: parameter update
        '''
        self.experience_maker.actor.update()
        self.experience_maker.critic.update()
        '''
        pass