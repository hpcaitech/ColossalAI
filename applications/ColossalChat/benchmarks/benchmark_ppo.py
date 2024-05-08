"""
For becnhmarking ppo. Mudified from examples/training_scripts/train_ppo.py
"""

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
    setup_distributed_dataloader,
)
from coati.models import Critic, RewardModel, convert_to_lora_module, disable_dropout
from coati.trainer import PPOTrainer
from coati.trainer.callbacks import PerformanceEvaluator
from coati.trainer.utils import is_rank_0
from coati.utils import load_checkpoint, replace_with_flash_attention
from transformers import AutoTokenizer, OPTForCausalLM
from transformers.models.opt.configuration_opt import OPTConfig

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device


def get_model_numel(model: torch.nn.Module, plugin: str, tp: int) -> int:
    numel = sum(p.numel() for p in model.parameters())
    if plugin == "3d" and tp > 1:
        numel *= dist.get_world_size()
    return numel


def get_gpt_config(model_name: str) -> OPTConfig:
    model_map = {
        "125m": OPTConfig.from_pretrained("facebook/opt-125m"),
        "350m": OPTConfig(hidden_size=1024, ffn_dim=4096, num_hidden_layers=24, num_attention_heads=16),
        "700m": OPTConfig(hidden_size=1280, ffn_dim=5120, num_hidden_layers=36, num_attention_heads=20),
        "1.3b": OPTConfig.from_pretrained("facebook/opt-1.3b"),
        "2.7b": OPTConfig.from_pretrained("facebook/opt-2.7b"),
        "3.5b": OPTConfig(hidden_size=3072, ffn_dim=12288, num_hidden_layers=32, num_attention_heads=32),
        "5.5b": OPTConfig(hidden_size=3840, ffn_dim=15360, num_hidden_layers=32, num_attention_heads=32),
        "6.7b": OPTConfig.from_pretrained("facebook/opt-6.7b"),
        "10b": OPTConfig(hidden_size=5120, ffn_dim=20480, num_hidden_layers=32, num_attention_heads=32),
        "13b": OPTConfig.from_pretrained("facebook/opt-13b"),
    }
    try:
        return model_map[model_name]
    except KeyError:
        raise ValueError(f'Unknown model "{model_name}"')


def benchmark_train(args):
    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch()
    coordinator = DistCoordinator()

    # ======================================================
    # Initialize Model, Objective, Optimizer and LR Scheduler
    # ======================================================
    init_ctx = LazyInitContext(default_device=get_current_device()) if "gemini" in args.plugin else nullcontext()

    booster_policy = None
    with init_ctx:
        actor = OPTForCausalLM(config=get_gpt_config(args.pretrain))
        # Disable dropout
        disable_dropout(actor)
        ref_model = OPTForCausalLM(config=get_gpt_config(args.pretrain))
        reward_model = RewardModel(config=get_gpt_config("350m"))
        critic = Critic(config=get_gpt_config("350m"))
        disable_dropout(critic)

        actor_numel = get_model_numel(actor, args.plugin, args.tp)
        critic_numel = get_model_numel(critic, args.plugin, args.tp)
        initial_model_numel = get_model_numel(ref_model, args.plugin, args.tp)
        reward_model_numel = get_model_numel(reward_model, args.plugin, args.tp)

        performance_evaluator = PerformanceEvaluator(
            actor_numel,
            critic_numel,
            initial_model_numel,
            reward_model_numel,
            enable_grad_checkpoint=False,
            ignore_episodes=2,
            train_config={"model": "facebook/opt-" + args.pretrain, "lora_rank": args.lora_rank, "plugin": args.plugin},
            save_path="./benchmark_performance_summarization.txt",
        )

        if args.tp > 1:
            if reward_model.model.config.architectures[0] != critic.model.config.architectures[0]:
                raise ValueError("Reward model and critic model must have the same architecture")
            if reward_model.model.config.architectures[0] == "BloomForCausalLM":
                from colossalai.shardformer.policies.bloom import BloomPolicy

                booster_policy = BloomPolicy()
            elif reward_model.model.config.architectures[0] == "LlamaForCausalLM":
                from colossalai.shardformer.policies.llama import LlamaPolicy

                booster_policy = LlamaPolicy()
            elif reward_model.model.config.architectures[0] == "GPT2LMHeadModel":
                from colossalai.shardformer.policies.gpt2 import GPT2Policy

                booster_policy = GPT2Policy()
            elif reward_model.model.config.architectures[0] == "ChatGLMModel":
                from colossalai.shardformer.policies.chatglm2 import ChatGLMPolicy

                booster_policy = ChatGLMPolicy()
            elif reward_model.model.config.architectures[0] == "OPTForCausalLM":
                from colossalai.shardformer.policies.opt import OPTPolicy

                booster_policy = OPTPolicy()
            else:
                raise ValueError("Unknown model architecture for policy")

        if args.lora_rank > 0:
            actor = convert_to_lora_module(actor, args.lora_rank, lora_train_bias=args.lora_train_bias)
            critic = convert_to_lora_module(critic, args.lora_rank, lora_train_bias=args.lora_train_bias)

    if args.grad_checkpoint and args.lora_rank == 0:
        actor.gradient_checkpointing_enable()
        critic.model.gradient_checkpointing_enable()
        coordinator.print_on_master(msg="Gradient checkpointing enabled successfully")
    elif args.lora_rank > 0:
        coordinator.print_on_master(msg="Gradient checkpointing will be disabled when LoRA is enabled")

    if args.use_flash_attn:
        replace_with_flash_attention(model=actor)
        replace_with_flash_attention(model=critic)
        coordinator.print_on_master(msg="Flash-attention enabled successfully")

    # configure tokenizer
    tokenizer_dir = args.tokenizer_dir if args.tokenizer_dir is not None else args.pretrain
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if os.path.exists(args.conversation_template_config):
        conversation_template_config = json.load(open(args.conversation_template_config, "r", encoding="utf8"))
        conversation_template = setup_conversation_template(
            tokenizer, chat_template_config=conversation_template_config, save_path=args.conversation_template_config
        )
        stop_token_ids = (
            conversation_template.assistant_line_end if len(conversation_template.assistant_line_end) > 0 else None
        )
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

    # configure dataset
    coordinator.print_on_master(f"Load dataset: {args.prompt_dataset}")
    mode_map = {"train": "train", "valid": "validation", "test": "test"}
    train_prompt_dataset = load_tokenized_dataset(dataset_paths=args.prompt_dataset, mode="train", mode_map=mode_map)
    coordinator.print_on_master(f"prompt dataset size: {len(train_prompt_dataset)}")
    data_collator = DataCollatorForPromptDataset(tokenizer=tokenizer, max_length=args.max_length - args.max_seq_len)
    train_prompt_dataloader = setup_distributed_dataloader(
        dataset=train_prompt_dataset,
        batch_size=args.experience_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=data_collator,
        use_tp=args.tp > 1,
    )

    if len(args.pretrain_dataset) > 0:
        train_pretrain_dataset = load_tokenized_dataset(
            dataset_paths=args.pretrain_dataset, mode="train", mode_map=mode_map
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, max_length=args.max_length)
        train_pretrain_dataloader = setup_distributed_dataloader(
            dataset=train_pretrain_dataset,
            batch_size=args.ptx_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=data_collator,
            use_tp=args.tp > 1,
        )
    else:
        train_pretrain_dataloader = None

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
    if args.plugin == "gemini":
        plugin = GeminiPlugin(
            precision=args.mixed_precision,
            initial_scale=2**16,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "gemini_auto":
        plugin = GeminiPlugin(
            precision=args.mixed_precision,
            placement_policy="auto",
            initial_scale=2**16,
            max_norm=args.grad_clip,
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
        plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=1,
            zero_stage=0,
            precision=args.mixed_precision,
        )
        custom_plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=1,
            zero_stage=0,
            precision=args.mixed_precision,
            custom_policy=booster_policy,
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    if args.plugin != "3d":
        custom_plugin = plugin

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
        stop_token_ids=stop_token_ids,
        kl_coef=args.kl_coef,
        ptx_coef=args.ptx_coef,
        train_batch_size=args.train_batch_size,
        buffer_limit=args.num_collect_steps * args.experience_batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_seq_len,
        use_cache=True,
        do_sample=True,
        temperature=0.7,
        accumulation_steps=args.accumulation_steps,
        save_dir=args.save_path,
        save_interval=args.save_interval,
        top_k=50,
        use_tp=args.tp > 1,
        offload_inference_models="gemini" not in args.plugin,
        callbacks=[performance_evaluator],
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

    if args.lora_rank > 0 and args.merge_lora_weights:
        from coati.models.lora import LORA_MANAGER

        # NOTE: set model to eval to merge LoRA weights
        LORA_MANAGER.merge_weights = True
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
    memory_consumption = torch.cuda.max_memory_allocated() / 1024**2
    if is_rank_0():
        with open("./benchmark_memory_consumption.txt", "a+") as f:
            f.write(
                f"Model=Opt-{args.pretrain}; lora_rank={args.lora_rank}; plugin={args.plugin}\nMax CUDA memory usage: {memory_consumption:.2f} MB\n"
            )
    coordinator.print_on_master(f"Max CUDA memory usage: {memory_consumption:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dataset", nargs="+", default=[])
    parser.add_argument("--pretrain_dataset", nargs="+", default=[])
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
    parser.add_argument("--pretrain", type=str, default=None)
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
    parser.add_argument("--ptx_batch_size", type=int, default=1)
    parser.add_argument("--lora_train_bias", type=str, default="none")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument("--merge_lora_weights", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=9e-6)
    parser.add_argument("--critic_lr", type=float, default=9e-6)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--ptx_coef", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--log_dir", default="logs", type=str)
    parser.add_argument("--use_wandb", default=False, action="store_true")
    parser.add_argument("--grad_checkpoint", default=False, action="store_true")
    parser.add_argument("--use_flash_attn", default=False, action="store_true")
    args = parser.parse_args()
    benchmark_train(args)
