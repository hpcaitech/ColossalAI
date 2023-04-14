from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from coati.experience_maker import Experience, ExperienceMaker
from coati.replay_buffer import ReplayBuffer
from coati.experience_maker import Experience
from save_vram_em import SaveVramExperienceMaker
from coati.models.base import Actor, Critic
from coati.models.generation_utils import update_model_kwargs_fn
from coati.models.loss import PolicyLoss, ValueLoss
from coati.replay_buffer import NaiveReplayBuffer
from torch.optim import Optimizer, lr_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from coati.trainer.callbacks.base import Callback
from coati.trainer.strategies.base import Strategy
from coati.trainer.utils import is_rank_0
from coati.models.utils import compute_reward, normalize
from transformers import AutoTokenizer
import wandb
import time
import torch.distributed as dist
from easy_dataset import EasyRewardDataset
import pandas as pd
import os
class RewardModelTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        loss_fn (callable): the loss function to use for training
        train_dataset (Dataset): the dataset to use for training
        valid_dataset (Dataset): the dataset to use for validation
        eval_dataset (Dataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
    """

    def __init__(
        self,
        model,
        strategy: Strategy,
        optim: Optimizer,
        loss_fn,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        eval_dataset: Dataset,
        batch_size: int = 1,
        max_epochs: int = 1,
        save_every_step: int = 1000,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        train_sampler = None

        if dist.is_initialized() and dist.get_world_size() > 1:
            train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=42, drop_last=True)
        self.train_dataloader = DataLoader(train_dataset,
                                           shuffle=(train_sampler is None),
                                           sampler=train_sampler,
                                           batch_size=batch_size,collate_fn=EasyRewardDataset.collate_fn)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,collate_fn=EasyRewardDataset.collate_fn)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True,collate_fn=EasyRewardDataset.collate_fn)

        self.model = strategy.setup_model(model)
        self.loss_fn = loss_fn
        self.optimizer = strategy.setup_optimizer(optim, self.model)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, self.train_dataloader.__len__() // 100)
        self.save_every_step = save_every_step

    def eval_acc(self, dataloader):
        dist = 0
        on = 0
        cnt = 0
        self.model.eval()
        with torch.no_grad():
            for chosen_ids, c_mask, reject_ids, r_mask in dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                reject_reward = self.model(reject_ids, attention_mask=r_mask)
                for i in range(len(chosen_reward)):
                    cnt += 1
                    if chosen_reward[i] > reject_reward[i]:
                        on += 1
                dist += (chosen_reward - reject_reward).mean().item()
            dist_mean = dist / len(dataloader)
            acc = on / cnt
        self.model.train()
        return dist_mean, acc

    def fit(self):
        #init a string with YYYY_MM_DD_HH_MM_SS
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
        epoch_bar = tqdm(range(self.epochs), desc='Train epoch', disable=not is_rank_0())
        for epoch in range(self.epochs):
            step_bar = tqdm(range(self.train_dataloader.__len__()),
                            desc='Train step of epoch %d' % epoch,
                            disable=not is_rank_0())
            # train
            self.model.train()
            cnt = 0
            acc = 0
            dist = 0
            for chosen_ids, c_mask, reject_ids, r_mask in self.train_dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                # print(f'0.cuda memory usage {torch.cuda.memory_allocated()/1024/1024}  MB')
                chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                # print(f'1.cuda memory usage {torch.cuda.memory_allocated()/1024/1024}  MB')
                reject_reward = self.model(reject_ids, attention_mask=r_mask)
                # print(f'1.cuda memory usage {torch.cuda.memory_allocated()/1024/1024}  MB')
                loss = self.loss_fn(chosen_reward, reject_reward)
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer)
                self.optimizer.zero_grad()
                cnt += 1
                if cnt == 100:
                    self.scheduler.step()
                    dist, acc = self.eval_acc(self.valid_dataloader)
                    cnt = 0
                    if is_rank_0():
                        log = pd.DataFrame([[step_bar.n, loss.item(), dist, acc]],
                                           columns=['step', 'loss', 'dist', 'acc'])
                        log_file = f'log_{time_str}.csv'
                        if not os.path.exists(log_file):
                            mode = 'w'
                        else:
                            mode = 'a'
                        log.to_csv(log_file, mode=mode, header=False, index=False)
                step_bar.update()
                step_bar.set_postfix({'dist': dist, 'acc': acc})

            # eval
            dist, acc = self.eval_acc(self.eval_dataloader)
            if is_rank_0():
                log = pd.DataFrame([[step_bar.n, loss.item(), dist, acc]], columns=['step', 'loss', 'dist', 'acc'])
                log_file = 'log.csv'
                if not os.path.exists(log_file):
                    mode = 'w'
                else:
                    mode = 'a'
                log.to_csv(log_file, mode=mode, header=False, index=False)
            epoch_bar.update()
            step_bar.set_postfix({'dist': dist, 'acc': acc})
            step_bar.close()

    def save_model(self,
                   path: str,
                   only_rank0: bool = False,
                   tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        self.strategy.save_model(model=self.model, path=path, only_rank0=only_rank0, tokenizer=tokenizer)


class Trainer(ABC):
    """
        Base class for rlhf trainers.

    Args:
        strategy (Strategy):the strategy to use for training
        experience_maker (ExperienceMaker): the experience maker to use for produce experience to fullfill replay buffer
        replay_buffer (ReplayBuffer): the replay buffer to use for training
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenizer (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        data_loader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    """

    def __init__(self,
                 strategy: Strategy,
                 experience_maker: ExperienceMaker,
                 replay_buffer: ReplayBuffer,
                 experience_batch_size: int = 8,
                 max_epochs: int = 1,
                 tokenizer: Optional[Callable[[Any], dict]] = None,
                 sample_replay_buffer: bool = False,
                 dataloader_pin_memory: bool = True,
                 callbacks: List[Callback] = [],
                 **generate_kwargs) -> None:
        super().__init__()
        self.strategy = strategy
        self.experience_maker = experience_maker
        self.replay_buffer = replay_buffer
        self.experience_batch_size = experience_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.sample_replay_buffer = sample_replay_buffer
        self.dataloader_pin_memory = dataloader_pin_memory
        self.callbacks = callbacks

    @abstractmethod
    def training_step(self, experience: Experience) -> Dict[str, Any]:
        pass

    def _make_experience(self, inputs: Union[Tensor, Dict[str, Tensor]]) -> Experience:
        if isinstance(inputs, Tensor):
            return self.experience_maker.make_experience(inputs, **self.generate_kwargs)
        elif isinstance(inputs, dict):
            return self.experience_maker.make_experience(**inputs, **self.generate_kwargs)
        else:
            raise ValueError(f'Unsupported input type "{type(inputs)}"')

    def _sample_prompts(self, prompts) -> list:
        indices = list(range(len(prompts)))
        sampled_indices = self.strategy.experience_sampler.choice(indices, self.experience_batch_size, replace=False)
        return [prompts[i] for i in sampled_indices]

    def _learn(self):
        # replay buffer may be empty at first, we should rebuild at each training
        if not self.sample_replay_buffer:
            dataloader = self.strategy.setup_dataloader(self.replay_buffer, self.dataloader_pin_memory)
            device = torch.cuda.current_device()
        if self.sample_replay_buffer:
            pbar = tqdm(range(self.max_epochs), desc='Train epoch', disable=not is_rank_0())
            for _ in pbar:
                experience = self.replay_buffer.sample()
                metrics = self.training_step(experience)
                pbar.set_postfix(metrics)
        else:
            for epoch in range(self.max_epochs):
                self._on_learn_epoch_start(epoch)
                if isinstance(dataloader.sampler, DistributedSampler):
                    dataloader.sampler.set_epoch(epoch)
                pbar = tqdm(dataloader, desc=f'Train epoch [{epoch+1}/{self.max_epochs}]', disable=not is_rank_0())
                for experience in pbar:
                    self._on_learn_batch_start()
                    experience.to_device(device)
                    metrics = self.training_step(experience)
                    self._on_learn_batch_end(metrics, experience)
                    pbar.set_postfix(metrics)
                self._on_learn_epoch_end(epoch)

    def fit(self,
            prompt_dataloader,
            pretrain_dataloader,
            num_episodes: int = 50000,
            max_timesteps: int = 500,
            update_timesteps: int = 5000) -> None:
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
        wandb.init(project="CHAT-PPO", name=time_str)
        wandb.watch(self.actor)
        timeppo = 0
        self.pretrain_dataloader = pretrain_dataloader
        self.prompt_dataloader = prompt_dataloader
        self._on_fit_start()
        for episode in range(num_episodes):
            self._on_episode_start(episode)
            for timestep in tqdm(range(max_timesteps),
                                 desc=f'Episode [{episode+1}/{num_episodes}]',
                                 disable=not is_rank_0()):
                timeppo += 1
                prompts = next(iter(self.prompt_dataloader))
                if isinstance(prompts, dict):
                    prompts = prompts['input_ids'].to(torch.cuda.current_device())
                self._on_make_experience_start()
                torch.cuda.empty_cache()
                # print(f'experience_maker cuda memeor usage now: {torch.cuda.memory_allocated(0)/1024/1024} MB')
                # self.experience_maker.initial_model.to(torch.cuda.current_device())
                # self.experience_maker.reward_model.to(torch.cuda.current_device())
                # print(f'experience_maker cuda memeor usage then: {torch.cuda.memory_allocated(0)/1024/1024} MB')
                experience = self._make_experience(prompts)
                self._on_make_experience_end(experience)
                self.replay_buffer.append(experience)
                if timeppo % update_timesteps == 0:
                    self.experience_maker.initial_model.to('cpu')
                    self.experience_maker.reward_model.to('cpu')
                    torch.cuda.empty_cache()
                    print(f'ppo cuda memeor usage now: {torch.cuda.memory_allocated(0)/1024/1024} MB')
                    print(f'ppo cuda memeor usage then: {torch.cuda.memory_allocated(0)/1024/1024} MB')
                    self._learn()
                    self.replay_buffer.clear()
                    self.actor.to('cpu')
                    self.critic.to('cpu')
            self._on_episode_end(episode)
        self._on_fit_end()
        wandb.finish()

    # TODO(ver217): maybe simplify these code using context
    def _on_fit_start(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_start()

    def _on_fit_end(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_end()

    def _on_episode_start(self, episode: int) -> None:
        for callback in self.callbacks:
            callback.on_episode_start(episode)

    def _on_episode_end(self, episode: int) -> None:
        for callback in self.callbacks:
            callback.on_episode_end(episode)

    def _on_make_experience_start(self) -> None:
        for callback in self.callbacks:
            callback.on_make_experience_start()

    def _on_make_experience_end(self, experience: Experience) -> None:
        for callback in self.callbacks:
            callback.on_make_experience_end(experience)

    def _on_learn_epoch_start(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_learn_epoch_start(epoch)

    def _on_learn_epoch_end(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_learn_epoch_end(epoch)

    def _on_learn_batch_start(self) -> None:
        for callback in self.callbacks:
            callback.on_learn_batch_start()

    def _on_learn_batch_end(self, metrics: dict, experience: Experience) -> None:
        for callback in self.callbacks:
            callback.on_learn_batch_end(metrics, experience)

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
        experience_maker = SaveVramExperienceMaker(actor, critic, reward_model, initial_model, kl_coef)
        replay_buffer = NaiveReplayBuffer(train_batch_size, buffer_limit, buffer_cpu_offload)
        generate_kwargs = _set_default_generate_kwargs(strategy, generate_kwargs, actor)
        super().__init__(strategy, experience_maker, replay_buffer, experience_batch_size, max_epochs, tokenizer,
                         sample_replay_buffer, dataloader_pin_memory, callbacks, **generate_kwargs)
        #two tokenizers will help to decouple different stages' model type and  tokenizer
        self.actor = actor
        self.critic = critic

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.ptx_coef = ptx_coef
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim

    def training_step(self, experience: Experience) -> Dict[str, float]:
        self.critic.cpu()
        self.actor.to(torch.cuda.current_device())
        self.actor.train()
        # policy loss
        num_actions = experience.action_mask.size(1)
        action_log_probs = self.actor(experience.sequences, num_actions, attention_mask=experience.attention_mask)
        actor_loss = self.actor_loss_fn(action_log_probs,
                                        experience.action_log_probs,
                                        experience.advantages,
                                        action_mask=experience.action_mask)

        # ptx loss
        if self.ptx_coef != 0:
            batch = next(iter(self.pretrain_dataloader))
            ptx = batch['input_ids'].to(torch.cuda.current_device())
            label = batch['labels'].to(torch.cuda.current_device())[:, 1:]
            attention_mask = batch['attention_mask'].to(torch.cuda.current_device())
            ptx_log_probs = self.actor.get_base_model()(ptx, attention_mask=attention_mask)['logits'][..., :-1, :]
            ptx_loss = self.ptx_loss_fn(ptx_log_probs.view(-1, ptx_log_probs.size(-1)), label.view(-1))
            actor_loss = ptx_loss * self.ptx_coef + actor_loss * (1 - self.ptx_coef)

        self.strategy.backward(actor_loss, self.actor, self.actor_optim)
        self.strategy.optimizer_step(self.actor_optim)
        self.actor_optim.zero_grad()
        self.actor.cpu()
        print(f'current cuda memory allocated after actor is optimized: {torch.cuda.memory_allocated()/1024**3:.2f}GB')
        # value loss
        self.critic.to(torch.cuda.current_device())
        self.critic.train()
        values = self.critic(experience.sequences,
                             action_mask=experience.action_mask,
                             attention_mask=experience.attention_mask)
        print("values",values) 
        critic_loss = self.critic_loss_fn(values,
                                          experience.values,
                                          experience.reward,
                                          action_mask=experience.action_mask)
        print("critic_loss",critic_loss)
        self.strategy.backward(critic_loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim)
        self.critic_optim.zero_grad()
        self.critic.cpu()
        print(f'current cuda memory allocated after critic is optimized: {torch.cuda.memory_allocated()/1024**3:.2f}GB')
        return {'reward': experience.reward.mean().item(),'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}
    
    def save_model(self, path: str, only_rank0: bool = False, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        self.strategy.save_model(model=self.actor, path=path, only_rank0=only_rank0, tokenizer=tokenizer)

    def save_model(self, path: str, only_rank0: bool = False, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        self.strategy.save_model(model=self.actor, path=path, only_rank0=only_rank0, tokenizer=tokenizer)


def _set_default_generate_kwargs(strategy: Strategy, generate_kwargs: dict, actor: Actor) -> None:
    origin_model = strategy._unwrap_actor(actor)
    new_kwargs = {**generate_kwargs}
    # use huggingface models method directly
    if 'prepare_inputs_fn' not in generate_kwargs and hasattr(origin_model, 'prepare_inputs_for_generation'):
        new_kwargs['prepare_inputs_fn'] = origin_model.prepare_inputs_for_generation

    if 'update_model_kwargs_fn' not in generate_kwargs:
        new_kwargs['update_model_kwargs_fn'] = update_model_kwargs_fn

    return new_kwargs
