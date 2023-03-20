from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from chatgpt.experience_maker import Experience, NaiveExperienceMaker
from chatgpt.models.base import Actor, Critic
from chatgpt.models.generation_utils import update_model_kwargs_fn
from chatgpt.models.loss import PolicyLoss, ValueLoss
from chatgpt.replay_buffer import DetachedReplayBuffer
from torch.optim import Optimizer

from .detached_base import DetachedTrainer
from .callbacks import Callback
from .strategies import Strategy

from .utils import is_rank_0

import ray

@ray.remote
class DetachedPPOTrainer(DetachedTrainer):
    '''
        Detached Trainer for PPO algorithm
    Args:
        strategy (Strategy): the strategy to use for training
        actor (Actor): the actor model in ppo algorithm
        critic (Critic): the critic model in ppo algorithm
        actor_optim (Optimizer): the optimizer to use for actor model
        critic_optim (Optimizer): the optimizer to use for critic model
        train_batch_size (int, defaults to 8): the batch size to use for training
        train_batch_size (int, defaults to 8): the batch size to use for training
        buffer_limit (int, defaults to 0): the max_size limitaiton of replay buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload replay buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        value_clip (float, defaults to 0.4): the clip coefficient of value loss
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    '''
    
    def __int__(self, 
                experience_maker_holder_name_list: List[str],
                strategy: Strategy,
                actor: Actor,
                critic: Critic,
                actor_optim: Optimizer,
                critic_optim: Optimizer,
                train_batch_size: int = 8,
                buffer_limit: int = 0,
                buffer_cpu_offload: bool = True,
                eps_clip: float = 0.2,
                value_clip: float = 0.4,
                experience_batch_size: int = 8,
                max_epochs: int = 1,
                tokenizer: Optional[Callable[[Any], dict]] = None,
                dataloader_pin_memory: bool = True,
                callbacks: List[Callback] = [],
                **generate_kwargs) -> None:
        detached_replay_buffer = DetachedReplayBuffer(train_batch_size, limit=buffer_limit, cpu_offload=buffer_cpu_offload)
        generate_kwargs = _set_default_generate_kwargs(strategy, generate_kwargs, actor)
        super().__init__(experience_maker_holder_name_list, strategy, detached_replay_buffer, experience_batch_size, max_epochs, tokenizer,
                         dataloader_pin_memory, callbacks, **generate_kwargs)
        self.actor = actor
        self.critic = critic
        
        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        
    def update_remote_makers(self):
        # TODO: balance duties
        if is_rank_0():
            self.update_target_holder_list(self.target_holder_name_list)
        for target_holder in self.target_holder_list:
            # TODO: reduce malloc
            with torch.no_grad():
                target_holder.update_experience_maker.remote(self.actor, self.critic)


    def training_step(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()
        self.critic.train()
        
        num_actions = experience.action_mask.size(1)
        action_log_probs = self.actor(experience.sequences, num_actions, attention_mask=experience.attention_mask)
        actor_loss = self.actor_loss_fn(action_log_probs,
                                        experience.action_log_probs,
                                        experience.advantages,
                                        action_mask=experience.action_mask)
        self.strategy.backward(actor_loss, self.actor, self.actor_optim)
        self.strategy.optimizer_step(self.actor_optim)
        self.actor_optim.zero_grad()
        
        values = self.critic(experience.sequences,
                             action_mask=experience.action_mask,
                             attention_mask=experience.attention_mask)
        critic_loss = self.critic_loss_fn(values,
                                          experience.values,
                                          experience.reward,
                                          action_mask=experience.action_mask)
        
        self.strategy.backward(critic_loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim)
        self.critic_optim.zero_grad()
        
        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}

def _set_default_generate_kwargs(strategy: Strategy, generate_kwargs: dict, actor: Actor) -> None:
    origin_model = strategy._unwrap_actor(actor)
    new_kwargs = {**generate_kwargs}
    # use huggingface models method directly
    if 'prepare_inputs_fn' not in generate_kwargs and hasattr(origin_model, 'prepare_inputs_for_generation'):
        new_kwargs['prepare_inputs_fn'] = origin_model.prepare_inputs_for_generation

    if 'update_model_kwargs_fn' not in generate_kwargs:
        new_kwargs['update_model_kwargs_fn'] = update_model_kwargs_fn

    return new_kwargs
