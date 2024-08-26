import argparse
import json
import os
import resource
from contextlib import nullcontext

import torch
import torch.distributed as dist
from coati.dataset import (
    DataCollatorForPromptDataset,
    DataCollatorForSupervisedDataset,
    StatefulDistributedSampler,
    load_tokenized_dataset,
    setup_conversation_template,
)
from coati.models import Critic, LoraConfig, RewardModel, convert_to_lora_module, disable_dropout, lora_manager
from coati.trainer import PPOTrainer
from coati.utils import load_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.shardformer.policies.auto_policy import get_autopolicy

logger = get_dist_logger()


def train(args):
    lora_config = None
    if args.lora_config is not None:
        lora_config = LoraConfig.from_file(args.lora_config)
    # check lora compatibility
    if "gemini" in args.plugin and lora_config is not None and lora_config.r > 0:
        raise ValueError("LoRA is not supported in GeminiPlugin. Please use other plugin")
    if args.plugin == "gemini_auto" and args.accumulation_steps > 1:
        raise ValueError("Gradient accumulation is not supported in GeminiPlugin. Please use other plugin")
    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch()
    coordinator = DistCoordinator()

    # ======================================================
    # Initialize Model, Objective, Optimizer and LR Scheduler
    # ======================================================
    # Temp Fix: Disable lazy init due to version conflict
    # init_ctx = (
    #     LazyInitContext(default_device=get_current_device()) if isinstance(plugin, (GeminiPlugin,)) else nullcontext()
    # )

    init_ctx = nullcontext()
    with init_ctx:
        if args.use_flash_attn:
            actor = AutoModelForCausalLM.from_pretrained(
                args.pretrain,
                torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
                use_flash_attention_2=True,
                local_files_only=True,
            )
            ref_model = AutoModelForCausalLM.from_pretrained(
                args.pretrain,
                torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
                use_flash_attention_2=True,
                local_files_only=True,
            )
            reward_model = RewardModel(
                args.rm_pretrain,
                torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
                use_flash_attention_2=True,
            )
            critic = Critic(
                args.rm_pretrain,
                torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
                use_flash_attention_2=True,
            )
            coordinator.print_on_master(msg="Flash-attention enabled successfully")
        else:
            actor = AutoModelForCausalLM.from_pretrained(args.pretrain, local_files_only=True)
            ref_model = AutoModelForCausalLM.from_pretrained(args.pretrain, local_files_only=True)
            reward_model = RewardModel(args.rm_pretrain)
            critic = Critic(args.rm_pretrain)

        if args.lora_config is not None:
            actor = convert_to_lora_module(actor, lora_config=lora_config)
            critic = convert_to_lora_module(critic, lora_config=lora_config)
            for name, module in actor.named_modules():
                if "norm" in name or "gate" in name:
                    module = module.to(torch.float32)
            for name, module in critic.named_modules():
                if "norm" in name or "gate" in name:
                    module = module.to(torch.float32)
            lora_manager.able_to_merge = False

        # Disable dropout
        disable_dropout(actor)
        disable_dropout(critic)

    if args.grad_checkpoint:
        actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        critic.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        coordinator.print_on_master(msg="Gradient checkpointing enabled successfully")

    # configure tokenizer
    tokenizer_dir = args.tokenizer_dir if args.tokenizer_dir is not None else args.pretrain
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False, trust_remote_code=True)
    if os.path.exists(args.conversation_template_config):
        with open(args.conversation_template_config, "r", encoding="utf8") as f:
            conversation_template_config = json.load(f)
        dist.barrier()
        conversation_template = setup_conversation_template(
            tokenizer, chat_template_config=conversation_template_config, save_path=args.conversation_template_config
        )
        stop_ids = conversation_template.stop_ids if len(conversation_template.stop_ids) > 0 else None
    else:
        raise ValueError("Conversation template config is not provided or incorrect")
    if hasattr(tokenizer, "pad_token") and hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
        try:
            # Some tokenizers doesn't allow to set pad_token mannually e.g., Qwen
            tokenizer.pad_token = tokenizer.eos_token
        except AttributeError as e:
            logger.warning(f"Unable to set pad token to eos token, {str(e)}")
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        logger.warning(
            "The tokenizer does not have a pad token which is required. May lead to unintended behavior in training, Please consider manually set them."
        )

    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    tokenizer.padding_side = "left"  # left padding for generation (online learning)

    # configure generation config
    actor.generation_config.update(
        pad_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id
    )

    # configure optimizer
    coordinator.print_on_master(f"setting up optimizer for actor: lr={args.lr}, weight_decay={args.weight_decay}")
    actor_optim = HybridAdam(
        model_params=actor.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        adamw_mode=True,
    )

    coordinator.print_on_master(f"setting up optimizer for critic: lr={args.lr}, weight_decay={args.weight_decay}")
    critic_optim = HybridAdam(
        model_params=critic.parameters(),
        lr=args.critic_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        adamw_mode=True,
    )

    if args.warmup_steps is None:
        args.warmup_steps = int(0.025 * args.num_episodes)
        coordinator.print_on_master(f"Warmup steps is set to {args.warmup_steps}")

    actor_lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=actor_optim,
        total_steps=args.num_episodes,
        warmup_steps=args.warmup_steps,
        eta_min=0.1 * args.lr,
    )

    critic_lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=critic_optim,
        total_steps=args.num_episodes,
        warmup_steps=args.warmup_steps,
        eta_min=0.1 * args.lr,
    )

    # ==============================
    # Initialize Booster
    # ==============================
    if args.plugin == "ddp":
        """
        Default torch ddp plugin without any acceleration, for
        debugging purpose acceleration, for debugging purpose
        """
        plugin = TorchDDPPlugin(find_unused_parameters=True)
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(
            precision=args.mixed_precision,
            placement_policy="static",
            initial_scale=2**16,
            max_norm=args.grad_clip,
            enable_gradient_accumulation=True,
            enable_flash_attention=args.use_flash_attn,
        )
    elif args.plugin == "gemini_auto":
        plugin = GeminiPlugin(
            precision=args.mixed_precision,
            placement_policy="auto",
            initial_scale=2**16,
            max_norm=args.grad_clip,
            enable_flash_attention=args.use_flash_attn,
        )
    elif args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=args.mixed_precision,
            initial_scale=2**16,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "zero2_cpu":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=args.mixed_precision,
            initial_scale=2**16,
            cpu_offload=True,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "3d":
        if args.use_flash_attn and (args.tp > 1 or args.pp > 1 or args.sp > 1 or args.enable_sequence_parallelism):
            logger.warning("Flash attention cannot be used with 3D parallelism for PPO training. Disabling it.")
            args.use_flash_attn = False
        plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=args.pp,
            sp_size=args.sp,
            sequence_parallelism_mode=args.sp_mode,
            zero_stage=args.zero_stage,
            enable_flash_attention=args.use_flash_attn,
            enable_sequence_parallelism=args.enable_sequence_parallelism,
            cpu_offload=True if args.zero_stage >= 1 and args.zero_cpu_offload else False,
            parallel_output=False,
            max_norm=args.grad_clip,
            precision=args.mixed_precision,
        )
        custom_plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=args.pp,
            sp_size=args.sp,
            sequence_parallelism_mode=args.sp_mode,
            zero_stage=args.zero_stage,
            enable_flash_attention=args.use_flash_attn,
            enable_sequence_parallelism=args.enable_sequence_parallelism,
            cpu_offload=True if args.zero_stage >= 1 and args.zero_cpu_offload else False,
            parallel_output=False,
            max_norm=args.grad_clip,
            precision=args.mixed_precision,
            custom_policy=get_autopolicy(reward_model.model),
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    if args.plugin != "3d":
        custom_plugin = plugin

    # configure dataset
    coordinator.print_on_master(f"Load dataset: {args.prompt_dataset}")
    mode_map = {"train": "train", "valid": "validation", "test": "test"}
    train_prompt_dataset = load_tokenized_dataset(dataset_paths=args.prompt_dataset, mode="train", mode_map=mode_map)
    data_collator = DataCollatorForPromptDataset(tokenizer=tokenizer, max_length=args.max_length - args.max_seq_len)

    train_prompt_dataloader = plugin.prepare_dataloader(
        dataset=train_prompt_dataset,
        batch_size=args.experience_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=data_collator,
        distributed_sampler_cls=StatefulDistributedSampler,
    )

    if len(args.ptx_dataset) > 0:
        train_ptx_dataset = load_tokenized_dataset(dataset_paths=args.ptx_dataset, mode="train", mode_map=mode_map)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, max_length=args.max_length)
        train_pretrain_dataloader = plugin.prepare_dataloader(
            dataset=train_ptx_dataset,
            batch_size=args.ptx_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=data_collator,
            distributed_sampler_cls=StatefulDistributedSampler,
        )
    else:
        train_pretrain_dataloader = None

    actor_booster = Booster(plugin=plugin)
    ref_booster = Booster(plugin=plugin)
    rm_booster = Booster(plugin=custom_plugin)
    critic_booster = Booster(plugin=custom_plugin)

    default_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    torch.set_default_dtype(default_dtype)
    actor, actor_optim, _, train_prompt_dataloader, actor_lr_scheduler = actor_booster.boost(
        model=actor,
        optimizer=actor_optim,
        lr_scheduler=actor_lr_scheduler,
        dataloader=train_prompt_dataloader,
    )

    critic, critic_optim, _, _, critic_lr_scheduler = critic_booster.boost(
        model=critic,
        optimizer=critic_optim,
        lr_scheduler=critic_lr_scheduler,
        dataloader=train_prompt_dataloader,
    )
    reward_model, _, _, _, _ = rm_booster.boost(model=reward_model, dataloader=train_prompt_dataloader)
    ref_model, _, _, _, _ = ref_booster.boost(model=ref_model, dataloader=train_prompt_dataloader)

    torch.set_default_dtype(torch.float)

    coordinator.print_on_master(f"Booster init max CUDA memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
    coordinator.print_on_master(
        f"Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB"
    )

    sampler_start_idx = 0
    start_step = 0

    if args.rm_checkpoint_path is not None:
        if "modeling" in args.rm_checkpoint_path:
            rm_booster.load_model(reward_model, args.rm_checkpoint_path)
        else:
            _, _, _ = load_checkpoint(
                load_dir=args.rm_checkpoint_path,
                booster=rm_booster,
                model=reward_model,
                optimizer=None,
                lr_scheduler=None,
            )
        coordinator.print_on_master(f"Loaded reward model checkpoint {args.rm_checkpoint_path}")

    if args.checkpoint_path is not None:
        if "modeling" in args.checkpoint_path:
            actor_booster.load_model(actor, args.checkpoint_path)
            ref_booster.load_model(ref_model, args.checkpoint_path)
            coordinator.print_on_master(f"Loaded actor and reference model {args.checkpoint_path}")
        else:
            _, start_step, sampler_start_idx = load_checkpoint(
                load_dir=args.checkpoint_path,
                booster=actor_booster,
                model=actor,
                optimizer=actor_optim,
                lr_scheduler=actor_lr_scheduler,
            )
            _, _, _ = load_checkpoint(
                load_dir=args.checkpoint_path,
                booster=ref_booster,
                model=ref_model,
                optimizer=critic_optim,
                lr_scheduler=critic_lr_scheduler,
            )
            assert isinstance(train_prompt_dataloader.sampler, StatefulDistributedSampler)
            train_prompt_dataloader.sampler.set_start_index(start_index=sampler_start_idx)

            coordinator.print_on_master(
                f"Loaded actor and reference model checkpoint {args.checkpoint_path} at spisode {start_step}"
            )
            coordinator.print_on_master(f"Loaded sample at index {sampler_start_idx}")

        coordinator.print_on_master(
            f"Checkpoint loaded max CUDA memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB"
        )
        coordinator.print_on_master(
            f"Checkpoint loaded CUDA memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB"
        )
        coordinator.print_on_master(
            f"Checkpoint loaded max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB"
        )

    if args.critic_checkpoint_path is not None:
        if "modeling" in args.critic_checkpoint_path:
            critic_booster.load_model(critic, args.critic_checkpoint_path)
        else:
            _, _, _ = load_checkpoint(
                load_dir=args.critic_checkpoint_path,
                booster=critic_booster,
                model=critic,
                optimizer=critic_optim,
                lr_scheduler=critic_lr_scheduler,
            )
        coordinator.print_on_master(f"Loaded critic checkpoint {args.critic_checkpoint_path}")
        coordinator.print_on_master(
            f"Checkpoint loaded max CUDA memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB"
        )
        coordinator.print_on_master(
            f"Checkpoint loaded CUDA memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB"
        )
        coordinator.print_on_master(
            f"Checkpoint loaded max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB"
        )

    # configure trainer
    trainer = PPOTrainer(
        actor_booster,
        critic_booster,
        actor,
        critic,
        reward_model,
        ref_model,
        actor_optim,
        critic_optim,
        actor_lr_scheduler,
        critic_lr_scheduler,
        tokenizer=tokenizer,
        stop_token_ids=stop_ids,
        kl_coef=args.kl_coef,
        ptx_coef=args.ptx_coef,
        train_batch_size=args.train_batch_size,
        buffer_limit=args.num_collect_steps * args.experience_batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_seq_len,
        use_cache=True,
        do_sample=True,
        temperature=0.7,
        apply_loss_mask=not args.disable_loss_mask,
        accumulation_steps=args.accumulation_steps,
        save_dir=args.save_path,
        save_interval=args.save_interval,
        top_k=50,
        use_tp=args.tp > 1,
        offload_inference_models="gemini" not in args.plugin,
        coordinator=coordinator,
    )

    trainer.fit(
        num_episodes=args.num_episodes,
        num_collect_steps=args.num_collect_steps,
        num_update_steps=args.num_update_steps,
        prompt_dataloader=train_prompt_dataloader,
        pretrain_dataloader=train_pretrain_dataloader,
        log_dir=args.log_dir,
        use_wandb=args.use_wandb,
    )

    if lora_config is not None and lora_config.r > 0:
        # NOTE: set model to eval to merge LoRA weights
        lora_manager.able_to_merge = True
        actor.eval()
        critic.eval()
    # save model checkpoint after fitting on only rank0
    coordinator.print_on_master("Start saving final actor model checkpoint")
    actor_booster.save_model(actor, os.path.join(trainer.actor_save_dir, "modeling"), shard=True)
    coordinator.print_on_master(
        f"Saved final actor model checkpoint at episodes {args.num_episodes} at folder {args.save_path}"
    )
    coordinator.print_on_master("Start saving final critic model checkpoint")
    critic_booster.save_model(critic, os.path.join(trainer.critic_save_dir, "modeling"), shard=True)
    coordinator.print_on_master(
        f"Saved final critic model checkpoint at episodes {args.num_episodes} at folder {args.save_path}"
    )
    coordinator.print_on_master(f"Max CUDA memory usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dataset", nargs="+", default=[])
    parser.add_argument("--ptx_dataset", nargs="+", default=[])
    parser.add_argument(
        "--plugin",
        type=str,
        default="gemini",
        choices=["gemini", "gemini_auto", "zero2", "zero2_cpu", "3d"],
        help="Choose which plugin to use",
    )
    parser.add_argument(
        "--conversation_template_config",
        type=str,
        default=None,
        help="Path \
        to save conversation template config files.",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--sp", type=int, default=1)
    parser.add_argument("--enable_sequence_parallelism", default=False, action="store_true")
    parser.add_argument("--zero_stage", type=int, default=0, help="Zero stage", choices=[0, 1, 2])
    parser.add_argument("--zero_cpu_offload", default=False, action="store_true")
    parser.add_argument("--sp_mode", type=str, default="split_gather", choices=["split_gather", "ring", "all_to_all"])
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--rm_pretrain", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--critic_checkpoint_path", type=str, default=None)
    parser.add_argument("--rm_checkpoint_path", type=str, help="Reward model checkpoint path")
    parser.add_argument("--save_path", type=str, default="actor_checkpoint_prompts")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--num_collect_steps", type=int, default=2)
    parser.add_argument("--num_update_steps", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--experience_batch_size", type=int, default=16)
    parser.add_argument("--ptx_batch_size", type=int, default=4)
    parser.add_argument("--lora_config", type=str, default=None, help="low-rank adaptation config file path")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=9e-6)
    parser.add_argument("--critic_lr", type=float, default=9e-6)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--ptx_coef", type=float, default=0.0)
    parser.add_argument("--disable_loss_mask", default=False, action="store_true")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--use_wandb", default=False, action="store_true")
    parser.add_argument("--grad_checkpoint", default=False, action="store_true")
    parser.add_argument("--use_flash_attn", default=False, action="store_true")
    args = parser.parse_args()
    train(args)
