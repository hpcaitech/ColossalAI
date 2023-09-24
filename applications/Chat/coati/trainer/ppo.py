from typing import Dict, List, Optional

from coati.experience_buffer import NaiveExperienceBuffer
from coati.experience_maker import Experience, NaiveExperienceMaker
from coati.models.base import Actor, Critic, RewardModel, get_base_model
from coati.models.loss import GPTLMLoss, PolicyLoss, ValueLoss
from coati.models.utils import calc_action_log_probs
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from colossalai.utils import get_current_device

from .base import OnPolicyTrainer
from .callbacks import Callback
from .strategies import GeminiStrategy, Strategy
from .utils import CycledDataLoader, is_rank_0, to_device


def _set_default_generate_kwargs(strategy: Strategy, generate_kwargs: dict, actor: Actor) -> Dict:
    unwrapped_model = strategy.unwrap_model(actor)
    hf_model = get_base_model(unwrapped_model)
    new_kwargs = {**generate_kwargs}
    # use huggingface models method directly
    if "prepare_inputs_fn" not in generate_kwargs and hasattr(hf_model, "prepare_inputs_for_generation"):
        new_kwargs["prepare_inputs_fn"] = hf_model.prepare_inputs_for_generation

    if "update_model_kwargs_fn" not in generate_kwargs and hasattr(hf_model, "_update_model_kwargs_for_generation"):
        new_kwargs["update_model_kwargs_fn"] = hf_model._update_model_kwargs_for_generation

    return new_kwargs


class PPOTrainer(OnPolicyTrainer):
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
        critic: Critic,
        reward_model: RewardModel,
        initial_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        tokenizer: PreTrainedTokenizerBase,
        kl_coef: float = 0.1,
        ptx_coef: float = 0.9,
        train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        vf_coef: float = 1.0,
        value_clip: float = 0.4,
        sample_buffer: bool = False,
        dataloader_pin_memory: bool = True,
        offload_inference_models: bool = True,
        callbacks: List[Callback] = [],
        **generate_kwargs,
    ) -> None:
        if isinstance(strategy, GeminiStrategy):
            assert not offload_inference_models, "GeminiPlugin is not compatible with manual model.to('cpu')"

        data_buffer = NaiveExperienceBuffer(train_batch_size, buffer_limit, buffer_cpu_offload)
        super().__init__(strategy, data_buffer, sample_buffer, dataloader_pin_memory, callbacks)

        self.generate_kwargs = _set_default_generate_kwargs(strategy, generate_kwargs, actor)
        self.experience_maker = NaiveExperienceMaker(actor, critic, reward_model, initial_model, tokenizer, kl_coef)

        self.actor = actor
        self.critic = critic
        self.tokenizer = tokenizer

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.vf_coef = vf_coef
        self.ptx_loss_fn = GPTLMLoss()
        self.ptx_coef = ptx_coef
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim

        self.offload_inference_models = offload_inference_models
        self.device = get_current_device()

    def _before_fit(
        self,
        prompt_dataloader: DataLoader,
        pretrain_dataloader: DataLoader,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
    ):
        """
        Args:
            prompt_dataloader (DataLoader): the dataloader to use for prompt data
            pretrain_dataloader (DataLoader): the dataloader to use for pretrain data
        """
        self.prompt_dataloader = CycledDataLoader(prompt_dataloader)
        self.pretrain_dataloader = CycledDataLoader(pretrain_dataloader)

        self.writer = None
        if use_wandb and is_rank_0():
            assert log_dir is not None, "log_dir must be provided when use_wandb is True"
            import wandb

            wandb.init(project="Coati-ppo", sync_tensorboard=True)
        if log_dir is not None and is_rank_0():
            import os
            import time

            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(log_dir, "ppo")
            log_dir = os.path.join(log_dir, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
            self.writer = SummaryWriter(log_dir=log_dir)

    def _make_experience(self, collect_step: int) -> Experience:
        prompts = self.prompt_dataloader.next()
        if self.offload_inference_models:
            # TODO(ver217): this may be controlled by strategy if they are prepared by strategy
            self.experience_maker.initial_model.to(self.device)
            self.experience_maker.reward_model.to(self.device)
        assert isinstance(prompts, dict), f'Unsupported input type "{type(prompts)}"'
        return self.experience_maker.make_experience(**prompts, **self.generate_kwargs)

    def _training_step(self, experience: Experience):
        self.actor.train()
        self.critic.train()
        # policy loss
        num_actions = experience.action_log_probs.size(1)
        actor_logits = self.actor(experience.sequences, experience.attention_mask)["logits"]
        action_log_probs = calc_action_log_probs(actor_logits, experience.sequences, num_actions)
        actor_loss = self.actor_loss_fn(
            action_log_probs, experience.action_log_probs, experience.advantages, action_mask=experience.action_mask
        )
        actor_loss = (1 - self.ptx_coef) * actor_loss
        self.strategy.backward(actor_loss, self.actor, self.actor_optim)

        # ptx loss
        if self.ptx_coef != 0:
            batch = self.pretrain_dataloader.next()
            batch = to_device(batch, self.device)
            ptx_log_probs = self.actor(batch["input_ids"], batch["attention_mask"])["logits"]
            ptx_loss = self.ptx_coef * self.ptx_loss_fn(ptx_log_probs, batch["labels"])
            self.strategy.backward(ptx_loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim)
        self.actor_optim.zero_grad()

        # value loss
        values = self.critic(experience.sequences, attention_mask=experience.attention_mask)
        critic_loss = self.critic_loss_fn(values, experience.values, experience.reward)
        critic_loss = critic_loss * self.vf_coef
        self.strategy.backward(critic_loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim)
        self.critic_optim.zero_grad()

    def _learn(self, update_step: int):
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
