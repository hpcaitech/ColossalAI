from typing import Any, Callable, Dict, List, Optional
import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim import Adam

from chatgpt.experience_maker import Experience, NaiveExperienceMaker
from chatgpt.models.base import Actor, Critic
from chatgpt.models.generation_utils import update_model_kwargs_fn
from chatgpt.models.loss import PolicyLoss, ValueLoss
from chatgpt.replay_buffer import DetachedReplayBuffer
from chatgpt.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy

from colossalai.nn.optimizer import HybridAdam

from .detached_base import DetachedTrainer
from .callbacks import Callback
from .strategies import Strategy
from .utils import is_rank_0, get_cuda_actor_critic_from_args

import ray


@ray.remote
class DetachedPPOTrainer(DetachedTrainer):
    '''
        Detached Trainer for PPO algorithm
    Args:
        strategy (Strategy): the strategy to use for training
        model (str) : for actor / critic init
        pretrained (str) : for actor / critic init
        lora_rank (int) : for actor / critic init
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

    def __init__(self,
                 experience_maker_holder_name_list: List[str],
                 strategy: Strategy,
                 model: str,
                 pretrained: str = None,
                 lora_rank: int = 0,
                 train_batch_size: int = 8,
                 buffer_limit: int = 0,
                 buffer_cpu_offload: bool = True,
                 eps_clip: float = 0.2,
                 value_clip: float = 0.4,
                 experience_batch_size: int = 8,
                 max_epochs: int = 1,
                 dataloader_pin_memory: bool = True,
                 callbacks: List[Callback] = [],
                 **generate_kwargs) -> None:
        self.fully_initialized = False
        
        self.strategy = strategy
        # configure models, loss and optimizers
        with self.strategy.model_init_context():
            self.actor, self.critic = get_cuda_actor_critic_from_args(model, pretrained, lora_rank)
        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        if isinstance(self.strategy, ColossalAIStrategy):
            self.actor_optim = HybridAdam(self.actor.parameters(), lr=5e-6)
            self.critic_optim = HybridAdam(self.critic.parameters(), lr=5e-6)
        else:
            self.actor_optim = Adam(self.actor.parameters(), lr=5e-6)
            self.critic_optim = Adam(self.critic.parameters(), lr=5e-6)
        (self.actor, self.actor_optim), (self.critic, self.critic_optim) = \
            self.strategy.prepare((self.actor, self.actor_optim), (self.critic, self.critic_optim))

        generate_kwargs = _set_default_generate_kwargs(strategy, generate_kwargs, self.actor)
        
        super().__init__(experience_maker_holder_name_list,
                         strategy=strategy,
                         train_batch_size=train_batch_size,
                         buffer_limit=buffer_limit,
                         buffer_cpu_offload=buffer_cpu_offload,
                         experience_batch_size=experience_batch_size,
                         max_epochs=max_epochs,
                         dataloader_pin_memory=dataloader_pin_memory,
                         callbacks=callbacks,
                         generate_kwargs=generate_kwargs)
        self.fully_initialized = True

    def update_remote_makers(self):
        # TODO: balance duties
        if is_rank_0():
            self.update_target_holder_list(self.target_holder_name_list)
        for target_holder in self.target_holder_list:
            # TODO: reduce malloc
            with torch.no_grad():
                target_holder.update_experience_maker.remote(self.actor, self.critic)

    def ready(self):
        # indicate that self is fully initialized
        while not hasattr(self, "fully_initialized") or self.fully_initialized == False:
            time.sleep(1.0)
        return True

    def training_step(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()
        self.critic.train()

        experience.to_device(torch.cuda.current_device())

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

    def strategy_save_actor(self, path: str, only_rank0: bool = False) -> None:
        self.strategy.save_model(self.actor, path, only_rank0)

    def strategy_save_critic(self, path: str, only_rank0: bool = False) -> None:
        self.strategy.save_model(self.critic, path, only_rank0)

    def strategy_save_actor_optim(self, path: str, only_rank0: bool = False) -> None:
        self.strategy.save_optimizer(self.actor_optim, path, only_rank0)

    def strategy_save_critic_optim(self, path: str, only_rank0: bool = False) -> None:
        self.strategy.save_optimizer(self.critic_optim, path, only_rank0)

    def get_actor(self):
        return self.actor
    
    def get_critic(self):
        return self.critic

def _set_default_generate_kwargs(strategy: Strategy, generate_kwargs: dict, actor: Actor) -> None:
    origin_model = strategy._unwrap_actor(actor)
    new_kwargs = {**generate_kwargs}
    # use huggingface models method directly
    if 'prepare_inputs_fn' not in generate_kwargs and hasattr(origin_model, 'prepare_inputs_for_generation'):
        new_kwargs['prepare_inputs_fn'] = origin_model.prepare_inputs_for_generation

    if 'update_model_kwargs_fn' not in generate_kwargs:
        new_kwargs['update_model_kwargs_fn'] = update_model_kwargs_fn

    return new_kwargs
