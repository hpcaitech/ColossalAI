from typing import Any, Callable, Dict, List, Optional, Tuple

import ray
import torch
from coati.experience_maker import Experience, NaiveExperienceMaker
from coati.models.base import Actor, Critic
from coati.models.loss import PolicyLoss, ValueLoss
from coati.trainer.callbacks import Callback
from coati.trainer.strategies import DDPStrategy, GeminiStrategy, LowLevelZeroStrategy, Strategy
from torch.optim import Adam

from colossalai.nn.optimizer import HybridAdam

from .callbacks import TrainerCallback, TrainerPerformanceEvaluator
from .detached_trainer_base import DetachedTrainer
from .lora_constructor import LoRAConstructor
from .utils import (
    get_actor_from_args,
    get_critic_from_args,
    get_model_numel,
    get_rank,
    get_strategy_from_args,
    is_rank_0,
    set_dist_env,
    state_dict_to,
)


@ray.remote(concurrency_groups={
    "buffer_length": 1,
    "buffer_append": 1,
    "buffer_sample": 1,
    "model_io": 1,
    "compute": 1
})
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
        buffer_limit (int, defaults to 0): the max_size limitation of replay buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload replay buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        value_clip (float, defaults to 0.4): the clip coefficient of value loss
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    '''

    def __init__(
        self,
        experience_maker_holder_name_list: List[str],
        strategy_fn: Callable[[], Strategy],
        model_fn: Callable[[], Tuple[Actor, Critic]],
        env_info: Dict[str, str] = None,
        train_batch_size: int = 8,
        buffer_limit: int = 0,
        eps_clip: float = 0.2,
        value_clip: float = 0.4,
        dataloader_pin_memory: bool = True,
        callbacks: List[TrainerCallback] = [],
        eval_performance: bool = False,
        debug: bool = False,
        update_lora_weights: bool = False,
    ) -> None:
        # set environment variables
        if env_info:
            set_dist_env(env_info=env_info)
        # configure strategy
        self.strategy = strategy_fn()
        # configure models, loss and optimizers
        with self.strategy.model_init_context():
            self.actor, self.critic = model_fn()

        if eval_performance:
            actor_numel = get_model_numel(self.actor)
            critic_numel = get_model_numel(self.critic)
            evaluator = TrainerPerformanceEvaluator(actor_numel, critic_numel)
            callbacks = callbacks + [evaluator]

        if isinstance(self.strategy, (LowLevelZeroStrategy, GeminiStrategy)):
            self.actor_optim = HybridAdam(self.actor.parameters(), lr=1e-7)
            self.critic_optim = HybridAdam(self.critic.parameters(), lr=1e-7)
        else:
            self.actor_optim = Adam(self.actor.parameters(), lr=1e-7)
            self.critic_optim = Adam(self.critic.parameters(), lr=1e-7)

        (self.actor, self.actor_optim), (self.critic, self.critic_optim) = \
            self.strategy.prepare((self.actor, self.actor_optim), (self.critic, self.critic_optim))

        # configure trainer
        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)

        super().__init__(experience_maker_holder_name_list,
                         train_batch_size=train_batch_size,
                         buffer_limit=buffer_limit,
                         dataloader_pin_memory=dataloader_pin_memory,
                         callbacks=callbacks,
                         debug=debug)
        if self._debug:
            print(f'[trainer{get_rank()}] will send state dict to {experience_maker_holder_name_list}')

        self._update_lora_weights = update_lora_weights

    @ray.method(concurrency_group="model_io")
    @torch.no_grad()
    def _update_remote_makers(self, fully_update: bool = False, **config):
        # TODO: balance duties
        if not fully_update:
            config['requires_grad_only'] = True
        self.update_target_holder_list()
        # mark start, ensure order
        tasks = []
        for target_holder in self.target_holder_list:
            tasks.append(target_holder.update_experience_maker.remote(chunk_start=True, fully_update=fully_update))
        ray.get(tasks)
        # sending loop
        tasks = []

        for state_dict_shard in self._get_model_state_dict_shard(self.actor, fully_update=fully_update, **config):
            for target_holder in self.target_holder_list:
                tasks.append(
                    target_holder.update_experience_maker.remote(
                        new_actor_state_dict=state_dict_shard,
                        new_actor_lora_config_dict=self._get_model_lora_config_dict(self.actor),
                        fully_update=fully_update))
        # sending loop
        for state_dict_shard in self._get_model_state_dict_shard(self.critic, fully_update=fully_update, **config):
            for target_holder in self.target_holder_list:
                tasks.append(
                    target_holder.update_experience_maker.remote(
                        new_critic_state_dict=state_dict_shard,
                        new_critic_lora_config_dict=self._get_model_lora_config_dict(self.critic),
                        fully_update=fully_update))
        ray.get(tasks)
        # mark end
        for target_holder in self.target_holder_list:
            target_holder.update_experience_maker.remote(chunk_end=True, fully_update=fully_update)

    @ray.method(concurrency_group="compute")
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

    def strategy_save_actor(self, path: str, only_rank0: bool = False) -> None:
        self.strategy.save_model(self.actor, path, only_rank0)

    def strategy_save_critic(self, path: str, only_rank0: bool = False) -> None:
        self.strategy.save_model(self.critic, path, only_rank0)

    def strategy_save_actor_optim(self, path: str, only_rank0: bool = False) -> None:
        self.strategy.save_optimizer(self.actor_optim, path, only_rank0)

    def strategy_save_critic_optim(self, path: str, only_rank0: bool = False) -> None:
        self.strategy.save_optimizer(self.critic_optim, path, only_rank0)

    def _get_model_state_dict_shard(self, model: torch.nn.Module, fully_update=False, **config):
        for state_dict in self.strategy.get_model_state_dict_shard(model, **config):
            if not self._update_lora_weights or fully_update:
                yield state_dict_to(state_dict)
            else:
                state_dict_lora, _ = LoRAConstructor.filter_state_dict_lora(state_dict)
                yield state_dict_to(state_dict_lora)

    def _get_model_lora_config_dict(self, model: torch.nn.Module):
        if not self._update_lora_weights:
            return None
        unwrapped_model = self.strategy.unwrap_model(model)
        return LoRAConstructor.extract_lora_config(unwrapped_model)
