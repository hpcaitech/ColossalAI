"""
GRPO trainer
"""

import os
from typing import Dict, List, Optional, Union

import torch
import wandb
from coati.experience_buffer import NaiveExperienceBuffer
from coati.experience_maker import Experience, NaiveExperienceMaker
from coati.models import RewardModel, RLVRRewardModel
from coati.models.loss import GPTLMLoss, PolicyLoss
from coati.models.utils import calc_action_log_probs
from coati.trainer.callbacks import Callback
from coati.trainer.utils import all_reduce_mean
from coati.utils import AccumulativeMeanMeter, save_checkpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.cluster import DistCoordinator
from colossalai.utils import get_current_device

from .base import OLTrainer
from .utils import AnnealingScheduler, CycledDataLoader, is_rank_0, to_device


def _set_default_generate_kwargs(actor: PreTrainedModel) -> Dict:
    """
    Set default keyword arguments for generation based on the actor model.

    Args:
        actor (PreTrainedModel): The actor model.

    Returns:
        Dict: A dictionary containing the default keyword arguments for generation.
    """
    unwrapped_model = actor.unwrap()
    new_kwargs = {}
    # use huggingface models method directly
    if hasattr(unwrapped_model, "prepare_inputs_for_generation"):
        new_kwargs["prepare_inputs_fn"] = unwrapped_model.prepare_inputs_for_generation
    if hasattr(unwrapped_model, "_update_model_kwargs_for_generation"):
        new_kwargs["update_model_kwargs_fn"] = unwrapped_model._update_model_kwargs_for_generation
    return new_kwargs


class GRPOTrainer(OLTrainer):
    """
        Trainer for GRPO algorithm.

    Args:
        strategy (Booster): the strategy to use for training
        actor (Actor): the actor model in ppo algorithm
        reward_model (RewardModel): the reward model in rlhf algorithm to make reward of sentences
        initial_model (Actor): the initial model in rlhf algorithm to generate reference logics to limit the update of actor
        actor_optim (Optimizer): the optimizer to use for actor model
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
        actor_booster: Booster,
        actor: PreTrainedModel,
        reward_model: Union[RewardModel, RLVRRewardModel],
        initial_model: PreTrainedModel,
        actor_optim: Optimizer,
        actor_lr_scheduler: _LRScheduler,
        tokenizer: PreTrainedTokenizerBase,
        kl_coef: float = 0.1,
        ptx_coef: float = 0.9,
        train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        vf_coef: float = 1.0,
        value_clip: float = 0.2,
        sample_buffer: bool = False,
        dataloader_pin_memory: bool = True,
        offload_inference_models: bool = True,
        apply_loss_mask: bool = True,
        accumulation_steps: int = 1,
        save_interval: int = 0,
        save_dir: str = None,
        use_tp: bool = False,
        num_generation: int = 8,
        inference_batch_size: int = None,
        logits_forward_batch_size: int = None,
        temperature_annealing_config: Optional[Dict] = None,
        coordinator: DistCoordinator = None,
        callbacks: List[Callback] = [],
        **generate_kwargs,
    ) -> None:
        if isinstance(actor_booster, GeminiPlugin):
            assert not offload_inference_models, "GeminiPlugin is not compatible with manual model.to('cpu')"

        data_buffer = NaiveExperienceBuffer(train_batch_size, buffer_limit, buffer_cpu_offload)
        super().__init__(actor_booster, None, data_buffer, sample_buffer, dataloader_pin_memory, callbacks=callbacks)
        self.generate_kwargs = _set_default_generate_kwargs(actor)
        self.generate_kwargs.update(generate_kwargs)

        self.actor = actor
        self.actor_booster = actor_booster
        self.actor_scheduler = actor_lr_scheduler
        self.tokenizer = tokenizer
        self.experience_maker = NaiveExperienceMaker(
            self.actor,
            None,
            reward_model,
            initial_model,
            self.tokenizer,
            kl_coef,
            use_grpo=True,
            num_generation=num_generation,
            inference_batch_size=inference_batch_size,
            logits_forward_batch_size=logits_forward_batch_size,
        )
        if temperature_annealing_config:
            # use annealing
            self.temperature_annealing_scheduler = AnnealingScheduler(
                temperature_annealing_config["start_temperature"],
                temperature_annealing_config["end_temperature"],
                temperature_annealing_config["annealing_warmup_steps"],
                temperature_annealing_config["annealing_steps"],
            )
        else:
            self.temperature_annealing_scheduler = None

        self.train_batch_size = train_batch_size

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.vf_coef = vf_coef
        self.ptx_loss_fn = GPTLMLoss()
        self.ptx_coef = ptx_coef
        self.actor_optim = actor_optim
        self.save_interval = save_interval
        self.apply_loss_mask = apply_loss_mask
        self.coordinator = coordinator
        self.actor_save_dir = os.path.join(save_dir, "actor")
        self.num_train_step = 0
        self.accumulation_steps = accumulation_steps
        self.use_tp = use_tp
        self.accumulative_meter = AccumulativeMeanMeter()
        self.offload_inference_models = offload_inference_models
        self.device = get_current_device()

    def _before_fit(
        self,
        prompt_dataloader: DataLoader,
        pretrain_dataloader: Optional[DataLoader] = None,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
    ):
        """
        Args:
            prompt_dataloader (DataLoader): the dataloader to use for prompt data
            pretrain_dataloader (DataLoader): the dataloader to use for pretrain data
        """
        self.prompt_dataloader = CycledDataLoader(prompt_dataloader)
        self.pretrain_dataloader = CycledDataLoader(pretrain_dataloader) if pretrain_dataloader is not None else None

        self.writer = None
        if use_wandb and is_rank_0():
            assert log_dir is not None, "log_dir must be provided when use_wandb is True"
            import wandb

            self.wandb_run = wandb.init(project="Coati-grpo", sync_tensorboard=True)
        if log_dir is not None and is_rank_0():
            import os
            import time

            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(log_dir, "grpo")
            log_dir = os.path.join(log_dir, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
            self.writer = SummaryWriter(log_dir=log_dir)

    def _setup_update_phrase_dataload(self):
        """
        why not use distributed_dataloader?
            if tp is used, input on each rank is the same and we use the same dataloader to feed same experience to all ranks
            if tp is not used, input on each rank is different and we expect different experiences to be fed to each rank
        """
        self.dataloader = DataLoader(
            self.data_buffer,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.data_buffer.collate_fn,
        )

    def _make_experience(self, collect_step: int) -> Experience:
        """
        Make experience
        """
        prompts = self.prompt_dataloader.next()
        if self.offload_inference_models:
            # TODO(ver217): this may be controlled by strategy if they are prepared by strategy
            self.experience_maker.initial_model.to(self.device)
            self.experience_maker.reward_model.to(self.device)
        if self.temperature_annealing_scheduler:
            self.generate_kwargs["temperature"] = self.temperature_annealing_scheduler.get_temperature()
        return self.experience_maker.make_experience(
            input_ids=prompts["input_ids"].to(get_current_device()),
            attention_mask=prompts["attention_mask"].to(get_current_device()),
            gt_answer=prompts["gt_answer"],
            **self.generate_kwargs,
        )

    def _training_step(self, experience: Experience):
        """
        Args:
            experience:
                sequences: [batch_size, prompt_length + response_length] --- <PAD>...<PAD><PROMPT>...<PROMPT><RESPONSE>...<RESPONSE><PAD>...<PAD>
        """
        self.actor.train()
        num_actions = experience.action_log_probs.size(1)
        # policy loss

        actor_logits = self.actor(input_ids=experience.sequences, attention_mask=experience.attention_mask)[
            "logits"
        ]  # [batch size, prompt_length + response_length]
        action_log_probs = calc_action_log_probs(actor_logits, experience.sequences, num_actions)
        actor_loss, to_skip, max_ratio = self.actor_loss_fn(
            action_log_probs,
            experience.action_log_probs,
            experience.advantages.unsqueeze(dim=-1).repeat_interleave(action_log_probs.size(-1), dim=-1),
            action_mask=experience.action_mask if self.apply_loss_mask else None,
        )
        # sequence that is not end properly are not counted in token cost
        token_cost = torch.sum(
            (experience.sequences[:, -num_actions:] != self.tokenizer.pad_token_id).to(torch.float), axis=-1
        ).to(actor_logits.device)
        end_properly = experience.sequences[:, -1] == self.tokenizer.pad_token_id
        mean_token_cost = torch.sum(token_cost * end_properly) / torch.sum(end_properly)
        actor_loss = (1 - self.ptx_coef) * actor_loss
        if not to_skip:
            self.actor_booster.backward(loss=actor_loss, optimizer=self.actor_optim)

        # ptx loss
        if self.ptx_coef != 0:
            batch = self.pretrain_dataloader.next()
            batch = to_device(batch, self.device)
            outputs = self.actor(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            ptx_loss = outputs.loss
            ptx_loss = self.ptx_coef * ptx_loss
            self.actor_booster.backward(loss=ptx_loss, optimizer=self.actor_optim)

        # sync
        actor_loss_mean = all_reduce_mean(tensor=actor_loss)
        max_ratio_mean = all_reduce_mean(tensor=max_ratio)
        reward_mean = all_reduce_mean(tensor=experience.reward.mean())
        advantages_mean = all_reduce_mean(tensor=experience.advantages.mean())
        kl_mean = all_reduce_mean(tensor=experience.kl.mean())
        mean_token_cost = all_reduce_mean(tensor=mean_token_cost)
        if self.ptx_coef != 0:
            ptx_loss_mean = all_reduce_mean(tensor=ptx_loss)

        self.accumulative_meter.add("actor_loss", actor_loss_mean.to(torch.float16).mean().item())
        self.accumulative_meter.add("max_ratio", max_ratio_mean.to(torch.float16).item())
        self.accumulative_meter.add("reward", reward_mean.to(torch.float16).mean().item())
        self.accumulative_meter.add("advantages", advantages_mean.to(torch.float16).item())
        self.accumulative_meter.add("skip_ratio", 1.0 if to_skip else 0.0)
        self.accumulative_meter.add("mean_token_cost", mean_token_cost.to(torch.float16).item())
        self.accumulative_meter.add("kl", kl_mean.to(torch.float16).item())
        if self.ptx_coef != 0:
            self.accumulative_meter.add("ptx_loss", ptx_loss_mean.to(torch.float16).mean().item())

        if self.num_train_step % self.accumulation_steps == self.accumulation_steps - 1:
            self.actor_optim.step()
            self.actor_optim.zero_grad()
            self.actor_scheduler.step()

            if self.temperature_annealing_scheduler:
                self.temperature_annealing_scheduler.step_forward()

            # preparing logging model output and corresponding rewards.
            if self.num_train_step % 10 == 0:
                response_text = self.experience_maker.tokenizer.batch_decode(
                    experience.sequences, skip_special_tokens=True
                )
                for i in range(len(response_text)):
                    response_text[i] = response_text[i] + f"\n\nReward: {experience.reward[i]}"

                if self.writer and is_rank_0() and "wandb_run" in self.__dict__:
                    # log output to wandb
                    my_table = wandb.Table(
                        columns=[f"sample response {i}" for i in range(len(response_text))], data=[response_text]
                    )
                    try:
                        self.wandb_run.log({"sample_response": my_table})
                    except OSError as e:
                        self.coordinator.print_on_master(e)
                elif self.writer and is_rank_0():
                    for line in response_text:
                        self.coordinator.print_on_master(line)

            if self.writer and is_rank_0():
                global_step = (self.num_train_step + 1) / self.accumulation_steps
                self.writer.add_scalar("train/max_ratio", self.accumulative_meter.get("max_ratio"), global_step)
                self.writer.add_scalar("train/skip_ratio", self.accumulative_meter.get("skip_ratio"), global_step)
                self.writer.add_scalar("train/actor_loss", self.accumulative_meter.get("actor_loss"), global_step)
                self.writer.add_scalar("train/lr_actor", self.actor_optim.param_groups[0]["lr"], global_step)
                if self.ptx_coef != 0:
                    self.writer.add_scalar("train/ptx_loss", self.accumulative_meter.get("ptx_loss"), global_step)
                self.writer.add_scalar("reward", self.accumulative_meter.get("reward"), global_step)
                self.writer.add_scalar("token_cost", self.accumulative_meter.get("mean_token_cost"), global_step)
                self.writer.add_scalar("approx_kl", self.accumulative_meter.get("kl"), global_step)
                self.writer.add_scalar("advantages", self.accumulative_meter.get("advantages"), global_step)
            self.accumulative_meter.reset()
        self.num_train_step += 1

    def _learn(self, update_step: int):
        """
        Perform the learning step of the PPO algorithm.

        Args:
            update_step (int): The current update step.

        Returns:
            None
        """
        if self.offload_inference_models:
            self.experience_maker.initial_model.to("cpu")
            self.experience_maker.reward_model.to("cpu")
        # buffer may be empty at first, we should rebuild at each training
        if self.sample_buffer:
            experience = self.data_buffer.sample()
            self._on_learn_batch_start()
            experience.to_device(self.device)
            self._training_step(experience)
            self._on_learn_batch_end(experience)
        else:
            if isinstance(self.dataloader.sampler, DistributedSampler):
                self.dataloader.sampler.set_epoch(update_step)
            pbar = tqdm(self.dataloader, desc=f"Train epoch [{update_step + 1}]", disable=not is_rank_0())
            for experience in pbar:
                self._on_learn_batch_start()
                experience.to_device(self.device)
                self._training_step(experience)
                self._on_learn_batch_end(experience)

    def _save_checkpoint(self, num_train_step: int = 0):
        """
        Save the actor checkpoints with running states.

        Args:
            num_train_step (int): The current num_train_step number.

        Returns:
            None
        """

        self.coordinator.print_on_master("\nStart saving actor checkpoint with running states")
        save_checkpoint(
            save_dir=self.actor_save_dir,
            booster=self.actor_booster,
            model=self.actor,
            optimizer=self.actor_optim,
            lr_scheduler=self.actor_scheduler,
            epoch=0,
            step=num_train_step + 1,
            batch_size=self.train_batch_size,
            coordinator=self.coordinator,
        )
        self.coordinator.print_on_master(
            f"Saved actor checkpoint at episode {(num_train_step + 1)} at folder {self.actor_save_dir}"
        )
