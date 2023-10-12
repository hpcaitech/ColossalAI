from typing import Optional

import torch
from coati.models.base import Actor
from coati.models.loss import GPTLMLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import PreTrainedTokenizerBase

from colossalai.utils import get_current_device

from .base import SLTrainer
from .strategies import Strategy
from .utils import is_rank_0


class PretrainTrainer(SLTrainer):
    """
        Trainer for PPO algorithm.

    Args:
        strategy (Strategy): the strategy to use for training
        actor (Actor): the actor model in ppo algorithm
        critic (Critic): the critic model in ppo algorithm
        reward_model (RewardModel): the reward model in rlhf algorithm to make reward of sentences
        initial_model (Actor): the initial model in rlhf algorithm to generate reference logics to limit the update of actor
        actor_optim (Optimizer): the optimizer to use for actor model
        critic_optim (Optimizer): the optimizer to use for critic model
        kl_coef (float, defaults to 0.1): the coefficient of kl divergence loss
        train_batch_size (int, defaults to 8): the batch size to use for training
        buffer_limit (int, defaults to 0): the max_size limitation of buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        vf_coef (float, defaults to 1.0): the coefficient of value loss
        ptx_coef (float, defaults to 0.9): the coefficient of ptx loss
        value_clip (float, defaults to 0.4): the clip coefficient of value loss
        sample_buffer (bool, defaults to False): whether to sample from buffer
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        offload_inference_models (bool, defaults to True): whether to offload inference models to cpu during training process
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    """

    def __init__(
        self,
        strategy: Strategy,
        actor: Actor,
        actor_optim: Optimizer,
        actor_lr_scheduler: _LRScheduler,
        tokenizer: PreTrainedTokenizerBase,
        max_epochs: int = 1,
    ) -> None:
        super().__init__(strategy=strategy, max_epochs=max_epochs, model=actor, optimizer=actor_optim)

        self.actor = actor
        self.actor_scheduler = actor_lr_scheduler
        self.tokenizer = tokenizer
        self.actor_loss_fn = GPTLMLoss()
        self.num_train_step = 0
        self.device = get_current_device()

    def _before_fit(
        self,
        train_preference_dataloader: DataLoader = None,
        eval_preference_dataloader: DataLoader = None,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
    ):
        """
        Args:
            prompt_dataloader (DataLoader): the dataloader to use for prompt data
            pretrain_dataloader (DataLoader): the dataloader to use for pretrain data
        """
        self.train_dataloader = train_preference_dataloader
        self.eval_dataloader = eval_preference_dataloader
        self.writer = None
        if use_wandb and is_rank_0():
            assert log_dir is not None, "log_dir must be provided when use_wandb is True"
            import wandb

            self.wandb_run = wandb.init(project="Coati-pretrain", sync_tensorboard=True)
        if log_dir is not None and is_rank_0():
            import os
            import time

            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(log_dir, "ppo")
            log_dir = os.path.join(log_dir, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
            self.writer = SummaryWriter(log_dir=log_dir)

    def _train(self, epoch: int):
        """
        Args:
            epoch int: the number of current epoch
        """
        self.actor.train()
        step_bar = trange(
            len(self.train_dataloader),
            desc=f"Epoch {epoch + 1}/{self.max_epochs}",
            disable=not is_rank_0(),
        )
        for i, batch in enumerate(self.train_dataloader):
            # print(self.tokenizer.batch_decode(batch[0], skip_special_tokens=True))
            # exit()
            input_ids, attention_mask = batch
            input_ids = input_ids.to(torch.cuda.current_device())
            attention_mask = attention_mask.to(torch.cuda.current_device())

            losses = self.actor(input_ids, attention_mask, calculate_loss=True)["loss"]

            loss = losses.mean()

            self.strategy.backward(loss, self.actor, self.optimizer)
            self.strategy.optimizer_step(self.optimizer)
            self.optimizer.zero_grad()
            self.actor_scheduler.step()

            if self.writer:
                self.writer.add_scalar("train/loss", loss, self.num_train_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.num_train_step)
            self.num_train_step += 1
            step_bar.update()
        step_bar.close()

    def _eval(self, epoch: int):
        """There is no evaluation stage for online learning model"""
