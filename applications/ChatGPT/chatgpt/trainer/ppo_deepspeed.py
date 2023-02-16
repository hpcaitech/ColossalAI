from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn
from chatgpt.experience_maker import Experience, NaiveExperienceMaker
from chatgpt.nn import Actor, Critic, PolicyLoss, ValueLoss
from chatgpt.replay_buffer import NaiveReplayBuffer
from torch.optim import Optimizer

from .base import Trainer
from .callbacks import Callback
from .strategies import Strategy


class DeepSpeedPPOTrainer(Trainer):

    def __init__(self,
                 strategy: Strategy,
                 actor: Actor,
                 critic: Critic,
                 reward_model: nn.Module,
                 initial_model: Actor,
                 kl_coef: float = 0.1,
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
        actor = Actor(actor.model)
        self.actor_engine, self.actor_optim = strategy.setup_model_and_optimizer(actor)
        self.critic_engine, self.critic_optim = strategy.setup_model_and_optimizer(critic)
        reward_model = strategy.setup_model(reward_model)
        initial_model = Actor(strategy.setup_model(initial_model.model))
        experience_maker = NaiveExperienceMaker(actor, critic, reward_model, initial_model, kl_coef)
        replay_buffer = NaiveReplayBuffer(train_batch_size, buffer_limit, buffer_cpu_offload)
        super().__init__(strategy, experience_maker, replay_buffer, experience_batch_size, max_epochs, tokenizer,
                         sample_replay_buffer, dataloader_pin_memory, callbacks, **generate_kwargs)

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)

    def training_step(self, experience: Experience) -> Dict[str, float]:
        self.actor_engine.train()
        self.actor_engine.train()

        num_actions = experience.action_mask.size(1)
        action_log_probs = self.actor_engine(experience.sequences,
                                             num_actions,
                                             attention_mask=experience.attention_mask)
        actor_loss = self.actor_loss_fn(action_log_probs,
                                        experience.action_log_probs,
                                        experience.advantages,
                                        action_mask=experience.action_mask)
        self.strategy.backward(actor_loss, self.actor_engine)
        self.strategy.optimizer_step(self.actor_engine)

        values = self.critic_engine(experience.sequences,
                                    action_mask=experience.action_mask,
                                    attention_mask=experience.attention_mask)
        critic_loss = self.critic_loss_fn(values,
                                          experience.values,
                                          experience.reward,
                                          action_mask=experience.action_mask)
        self.strategy.backward(critic_loss, self.critic_engine)
        self.strategy.optimizer_step(self.critic_engine)

        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}
