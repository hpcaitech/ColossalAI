from typing import Optional

import torch
from coati.models.base import Actor
from coati.models.loss import DpoLoss
from coati.models.utils import calc_masked_log_probs
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import PreTrainedTokenizerBase

from colossalai.utils import get_current_device

from .base import SLTrainer
from .strategies import Strategy
from .utils import is_rank_0


class DPOTrainer(SLTrainer):
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
        ref_model: Actor,
        actor_optim: Optimizer,
        actor_lr_scheduler: _LRScheduler,
        tokenizer: PreTrainedTokenizerBase,
        max_epochs: int = 1,
        beta: float = 0.1,
        disable_reference: bool = False,
    ) -> None:
        super().__init__(strategy=strategy, max_epochs=max_epochs, model=actor, optimizer=actor_optim)

        self.actor = actor
        self.ref_model = ref_model
        self.actor_scheduler = actor_lr_scheduler
        self.tokenizer = tokenizer
        self.actor_loss_fn = DpoLoss(beta)
        self.num_train_step = 0
        self.disable_reference = disable_reference
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

            self.wandb_run = wandb.init(project="Coati-dpo", sync_tensorboard=True)
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
        self.ref_model.eval()
        step_bar = trange(
            len(self.train_dataloader),
            desc=f"Epoch {epoch + 1}/{self.max_epochs}",
            disable=not is_rank_0(),
        )
        for i, batch in enumerate(self.train_dataloader):
            # print(self.tokenizer.batch_decode(batch[0], skip_special_tokens=True))
            # exit()
            chosen_input_ids, chosen_attention_mask, reject_input_ids, reject_attention_mask = batch
            # print(chosen_input_ids[0])
            # print('\n\n')
            # print(self.tokenizer.batch_decode(chosen_input_ids, skip_special_tokens=False)[0])
            # print("\n\n")
            # print(chosen_attention_mask[0])
            # print('\n\n')
            chosen_input_ids = chosen_input_ids.to(torch.cuda.current_device())
            chosen_attention_mask = chosen_attention_mask.to(torch.cuda.current_device())
            reject_input_ids = reject_input_ids.to(torch.cuda.current_device())
            reject_attention_mask = reject_attention_mask.to(torch.cuda.current_device())
            chosen_mask = chosen_attention_mask.clone()
            reject_mask = reject_attention_mask.clone()
            first_diff_position = torch.argmax((chosen_input_ids != reject_input_ids).float(), dim=1)
            for i in range(chosen_mask.size(0)):
                chosen_mask[i, : first_diff_position[i]] = False
                reject_mask[i, : first_diff_position[i]] = False

            # print(chosen_input_ids[0])
            # print(reject_input_ids[0])
            # set the mask value correspond to the first padding token to 1

            actor_chosen_logits = self.actor(chosen_input_ids, chosen_attention_mask)["logits"]
            actor_reject_logits = self.actor(reject_input_ids, reject_attention_mask)["logits"]

            # print(self.tokenizer.batch_decode(chosen_input_ids * chosen_mask, skip_special_tokens=False)[0])
            # print("\n\n")
            # print(chosen_mask[0])
            # print('\n\n')
            logprob_actor_chosen = calc_masked_log_probs(actor_chosen_logits, chosen_input_ids, chosen_mask[:, 1:])
            # print(logprob_actor_chosen[0])
            # print("\n\n")
            logprob_actor_reject = calc_masked_log_probs(actor_reject_logits, reject_input_ids, reject_mask[:, 1:])
            if not self.disable_reference:
                with torch.no_grad():
                    ref_chosen_logits = self.ref_model(chosen_input_ids, chosen_attention_mask)["logits"]
                    ref_reject_logits = self.ref_model(reject_input_ids, reject_attention_mask)["logits"]
                    logprob_ref_chosen = calc_masked_log_probs(ref_chosen_logits, chosen_input_ids, chosen_mask[:, 1:])
                    logprob_ref_reject = calc_masked_log_probs(ref_reject_logits, reject_input_ids, reject_mask[:, 1:])
            else:
                logprob_ref_chosen = None
                logprob_ref_reject = None
            # print(logprob_ref_chosen[0])
            # print("\n\n")
            # exit()
            losses, chosen_rewards, rejected_rewards = self.actor_loss_fn(
                logprob_actor_chosen.sum(-1) / chosen_mask[:, 1:].float().sum(-1),
                logprob_actor_reject.sum(-1) / reject_mask[:, 1:].float().sum(-1),
                logprob_ref_chosen.sum(-1) / chosen_mask[:, 1:].float().sum(-1)
                if logprob_ref_chosen is not None
                else None,
                logprob_ref_reject.sum(-1) / reject_mask[:, 1:].float().sum(-1)
                if logprob_ref_reject is not None
                else None,
            )
            # print(chosen_rewards[0])
            # print(rejected_rewards[0])
            # exit()
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            loss = losses.mean()
            self.strategy.backward(loss, self.actor, self.optimizer)
            self.strategy.optimizer_step(self.optimizer)
            self.optimizer.zero_grad()
            self.actor_scheduler.step()
            # print((losses*loss_mask)[0])
            # exit()

            if self.writer:
                self.writer.add_scalar("train/loss", loss, self.num_train_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.num_train_step)
                self.writer.add_scalar("train/chosen_rewards", chosen_rewards.mean(), self.num_train_step)
                self.writer.add_scalar(
                    "train/rejected_rewards",
                    rejected_rewards.mean(),
                    self.num_train_step,
                )
                self.writer.add_scalar(
                    "train/accuracy",
                    reward_accuracies.mean(),
                    self.num_train_step,
                )
            self.num_train_step += 1
            step_bar.update()
        step_bar.close()

    def _eval(self, epoch: int):
        """There is no evaluation stage for online learning model"""
