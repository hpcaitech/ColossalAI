from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from coati.experience_maker import Experience, NaiveExperienceMaker
from coati.models.base import Actor, Critic
from coati.models.generation_utils import update_model_kwargs_fn
from coati.models.loss import PolicyLoss, ValueLoss
from coati.replay_buffer import NaiveReplayBuffer
from torch.optim import Optimizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .base import Trainer
from .callbacks import Callback
from .strategies import Strategy


class PPOTrainer(Trainer):
    """
        Trainer for PPO algorithm.

    Args:
        strategy (Strategy): the strategy to use for training
        actor (Actor): the actor model in ppo algorithm
        critic (Critic): the critic model in ppo algorithm
        reward_model (nn.Module): the reward model in rlhf algorithm to make reward of sentences
        initial_model (Actor): the initial model in rlhf algorithm to generate reference logits to limit the update of actor
        actor_optim (Optimizer): the optimizer to use for actor model
        critic_optim (Optimizer): the optimizer to use for critic model
        kl_coef (float, defaults to 0.1): the coefficient of kl divergence loss
        train_batch_size (int, defaults to 8): the batch size to use for training
        buffer_limit (int, defaults to 0): the max_size limitaiton of replay buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload replay buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        value_clip (float, defaults to 0.4): the clip coefficient of value loss
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenier (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    """

    def __init__(self,
                 strategy: Strategy,
                 actor: Actor,
                 critic: Critic,
                 reward_model: nn.Module,
                 initial_model: Actor,
                 actor_optim: Optimizer,
                 critic_optim: Optimizer,
                 kl_coef: float = 0.1,
                 ptx_coef: float = 0.9,
                 train_batch_size: int = 8,
                 buffer_limit: int = 0,
                 buffer_cpu_offload: bool = True,
                 eps_clip: float = 0.2,
                 value_clip: float = 0.4,
                 experience_batch_size: int = 8,
                 max_epochs: int = 1,
                 tokenizer: Optional[Callable[[Any], dict]] = None,
                 sample_replay_buffer: bool = False,
                 dataloader_pin_memory: bool = True,
                 callbacks: List[Callback] = [],
                 **generate_kwargs) -> None:
        experience_maker = NaiveExperienceMaker(actor, critic, reward_model, initial_model, kl_coef)
        replay_buffer = NaiveReplayBuffer(train_batch_size, buffer_limit, buffer_cpu_offload)
        generate_kwargs = _set_default_generate_kwargs(strategy, generate_kwargs, actor)
        super().__init__(strategy, experience_maker, replay_buffer, experience_batch_size, max_epochs, tokenizer,
                         sample_replay_buffer, dataloader_pin_memory, callbacks, **generate_kwargs)
        self.actor = actor
        self.critic = critic

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.ptx_coef = ptx_coef
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim

    def training_step(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()
        self.critic.train()
        # policy loss
        num_actions = experience.action_mask.size(1)
        action_log_probs = self.actor(experience.sequences, num_actions, attention_mask=experience.attention_mask)
        actor_loss = self.actor_loss_fn(action_log_probs,
                                        experience.action_log_probs,
                                        experience.advantages,
                                        action_mask=experience.action_mask)

        # ptx loss
        if self.ptx_coef != 0:
            ptx = next(iter(self.pretrain_dataloader))['input_ids'].to(torch.cuda.current_device())
            label = next(iter(self.pretrain_dataloader))['labels'].to(torch.cuda.current_device())[:, 1:]
            attention_mask = next(iter(self.pretrain_dataloader))['attention_mask'].to(torch.cuda.current_device())
            ptx_log_probs = self.actor.get_base_model()(ptx, attention_mask=attention_mask)['logits'][..., :-1, :]
            ptx_loss = self.ptx_loss_fn(ptx_log_probs.view(-1, ptx_log_probs.size(-1)), label.view(-1))
            actor_loss = ptx_loss * self.ptx_coef + actor_loss * (1 - self.ptx_coef)

        self.strategy.backward(actor_loss, self.actor, self.actor_optim)
        self.strategy.optimizer_step(self.actor_optim)
        self.actor_optim.zero_grad()

        # value loss
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

        return {'reward': experience.reward.mean().item()}


def _set_default_generate_kwargs(strategy: Strategy, generate_kwargs: dict, actor: Actor) -> None:
    origin_model = strategy._unwrap_actor(actor)
    new_kwargs = {**generate_kwargs}
    # use huggingface models method directly
    if 'prepare_inputs_fn' not in generate_kwargs and hasattr(origin_model, 'prepare_inputs_for_generation'):
        new_kwargs['prepare_inputs_fn'] = origin_model.prepare_inputs_for_generation

    if 'update_model_kwargs_fn' not in generate_kwargs:
        new_kwargs['update_model_kwargs_fn'] = update_model_kwargs_fn

    return new_kwargs


def save_model(self, path: str, only_rank0: bool = False, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
    self.strategy.save_model(model=self.actor, path=path, only_rank0=only_rank0, tokenizer=tokenizer)
