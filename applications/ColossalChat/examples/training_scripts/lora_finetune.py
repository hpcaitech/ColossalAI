#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised fine-tuning of MoE models like Deepseek V3/R1 on a downstream task.
"""

import argparse
import json
import os
import resource
from contextlib import nullcontext
from types import MethodType

import torch
import torch.distributed as dist
from coati.dataset.loader import RawConversationDataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import (
    GeminiPlugin,
    HybridParallelPlugin,
    LowLevelZeroPlugin,
    MoeHybridParallelPlugin,
    Plugin,
    TorchDDPPlugin,
)
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device


def all_reduce_mean(loss: torch.Tensor, plugin: Plugin) -> torch.Tensor:
    loss = loss.data
    group = getattr(plugin, "dp_group", None)
    dist.all_reduce(loss, group=group)
    return loss / dist.get_world_size(group)


def train(args) -> None:
    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch()
    accelerator = get_accelerator()
    coordinator = DistCoordinator()

    # ==============================
    # Initialize Booster
    # ==============================
    if args.plugin == "ddp":
        plugin = TorchDDPPlugin(find_unused_parameters=True if args.use_grad_checkpoint is False else False)
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(
            precision=args.mixed_precision,
            initial_scale=2**16,
            max_norm=args.grad_clip,
            enable_gradient_accumulation=(args.accumulation_steps > 1),
            enable_fused_normalization=get_accelerator().is_available(),
            enable_flash_attention=args.use_flash_attn,
        )
    elif args.plugin == "gemini_auto":
        plugin = GeminiPlugin(
            precision=args.mixed_precision,
            placement_policy="auto",
            initial_scale=2**16,
            max_norm=args.grad_clip,
            enable_gradient_accumulation=(args.accumulation_steps > 1),
            enable_fused_normalization=get_accelerator().is_available(),
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
        plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=args.pp,
            sp_size=args.sp,
            sequence_parallelism_mode=args.sp_mode,
            zero_stage=args.zero_stage,
            enable_flash_attention=args.use_flash_attn,
            enable_fused_normalization=get_accelerator().is_available(),
            enable_sequence_parallelism=args.enable_sequence_parallelism,
            cpu_offload=True if args.zero_stage >= 1 and args.zero_cpu_offload else False,
            max_norm=args.grad_clip,
            precision=args.mixed_precision,
            microbatch_size=args.microbatch_size,
        )
    elif args.plugin == "moe":
        plugin = MoeHybridParallelPlugin(
            ep_size=args.ep,
            tp_size=args.tp,
            pp_size=args.pp,
            zero_stage=args.zero_stage,
            sp_size=args.sp,
            sequence_parallelism_mode=args.sp_mode,
            enable_sequence_parallelism=args.sp > 1,
            enable_fused_normalization=get_accelerator().is_available(),
            enable_flash_attention=args.use_flash_attn,
            max_norm=args.grad_clip,
            precision=args.mixed_precision,
            microbatch_size=args.microbatch_size,
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    booster = Booster(plugin=plugin)

    def is_master():
        if isinstance(plugin, HybridParallelPlugin) and plugin.pp_size > 1:
            return coordinator.rank == coordinator.world_size - 1
        return coordinator.is_master()

    # ==============================
    # Initialize Tensorboard and Save Config
    # ==============================
    if is_master():
        if args.tensorboard_dir is not None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(args.tensorboard_dir, exist_ok=True)
            writer = SummaryWriter(args.tensorboard_dir)

        with open(args.config_file, "w") as f:
            json.dump(args.__dict__, f, indent=4)

    # ======================================================
    # Initialize Tokenizer, Dataset, Collator and Dataloader
    # ======================================================
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, trust_remote_code=True)

    coordinator.print_on_master(
        f"Training Info:\nConfig file: {args.config_file} \nTensorboard logs: {args.tensorboard_dir} \nModel checkpoint: {args.save_dir}"
    )

    coordinator.print_on_master(f"Load dataset: {args.dataset}")
    dataset = RawConversationDataset(
        tokenizer,
        args.dataset,
        args.max_length,
    )

    dataloader = plugin.prepare_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    coordinator.print_on_master(
        f"Max device memory after data loader: {accelerator.max_memory_allocated() / 1024 ** 2:.2f} MB"
    )

    # ======================================================
    # Initialize Model, Objective, Optimizer and LR Scheduler
    # ======================================================
    # When training the ChatGLM model, LoRA and gradient checkpointing are incompatible.
    init_ctx = (
        LazyInitContext(default_device=get_current_device())
        if isinstance(plugin, (GeminiPlugin, HybridParallelPlugin))
        else nullcontext()
    )
    attn_impl = "eager" if get_accelerator().name == "npu" else "flash_attention_2"

    config = AutoConfig.from_pretrained(args.pretrained, trust_remote_code=True)

    with init_ctx:
        # from_pretrained is not compatible with LoRA, we load pretrained weights later.
        # model = AutoModelForCausalLM.from_pretrained(
        #     args.pretrained,
        #     torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
        #     trust_remote_code=True,
        #     attn_implementation=attn_impl,
        # )
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
        )

        if args.lora_rank > 0:
            if model.__class__.__name__.startswith("DeepseekV3"):
                lora_config = LoraConfig(
                    task_type="CAUSAL_LM",
                    r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    target_modules=["gate_proj", "up_proj", "down_proj"],
                )
            else:
                lora_config = LoraConfig(task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=args.lora_alpha)
            model = booster.enable_lora(model, lora_config=lora_config)

    # this is essential, otherwise the grad checkpoint will not work.
    model.train()

    if args.use_grad_checkpoint:
        model.gradient_checkpointing_enable()
        coordinator.print_on_master(msg="Gradient checkpointing enabled successfully")
    if model.config.__class__.__name__.startswith("DeepseekV3"):
        model.config.use_cache = False
        model.eval()
        # enable grad for moe layers
        for m in model.modules():
            if m.__class__.__name__ == "DeepseekV3MoE":
                m.moe_infer = MethodType(m.moe_infer.__wrapped__, m)

    model_numel = sum(p.numel() for p in model.parameters())
    coordinator.print_on_master(f"Model params: {model_numel / 1e9:.2f} B")

    optimizer = HybridAdam(
        model_params=model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        adamw_mode=True,
    )

    if args.warmup_steps is None:
        args.warmup_steps = int(args.num_epochs * 0.025 * (len(dataloader) // args.accumulation_steps))
        coordinator.print_on_master(f"Warmup steps is set to {args.warmup_steps}")

    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer,
        total_steps=args.num_epochs * (len(dataloader) // args.accumulation_steps),
        warmup_steps=args.warmup_steps,
        eta_min=0.1 * args.lr,
    )

    # Flash attention will be disabled because it does NOT support fp32.
    default_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    torch.set_default_dtype(default_dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )

    torch.set_default_dtype(torch.float)
    booster.load_model(model, args.pretrained, low_cpu_mem_mode=False, num_threads=8)

    coordinator.print_on_master(
        f"Booster init max device memory: {accelerator.max_memory_allocated() / 1024 ** 2:.2f} MB"
    )
    coordinator.print_on_master(
        f"Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB"
    )

    start_epoch = 0
    start_step = 0

    num_steps_per_epoch = len(dataloader) // args.accumulation_steps

    for epoch in range(start_epoch, args.num_epochs):
        dataloader.sampler.set_epoch(epoch=epoch)
        if isinstance(plugin, HybridParallelPlugin) and plugin.pp_size > 1:
            data_iter = iter(dataloader)
            step_bar = tqdm(
                range(len(dataloader)),
                desc="Step",
                disable=not is_master(),
            )
            for step in step_bar:
                outputs = booster.execute_pipeline(
                    data_iter,
                    model,
                    criterion=lambda outputs, inputs: outputs[0],
                    optimizer=optimizer,
                    return_loss=True,
                )
                loss = outputs["loss"]
                if booster.plugin.stage_manager.is_last_stage():
                    global_loss = all_reduce_mean(loss, plugin)

                optimizer.step()

                if booster.plugin.stage_manager.is_last_stage():
                    grad_norm = optimizer.get_grad_norm()
                    step_bar.set_postfix({"loss": global_loss.item(), "grad_norm": grad_norm})

                if args.tensorboard_dir is not None and is_master():
                    global_step = (epoch * num_steps_per_epoch) + (step + 1) // args.accumulation_steps
                    writer.add_scalar(tag="Loss", scalar_value=global_loss.item(), global_step=global_step)
                    writer.add_scalar(
                        tag="Learning Rate",
                        scalar_value=lr_scheduler.get_last_lr()[0],
                        global_step=global_step,
                    )
                    writer.add_scalar(tag="Grad Norm", scalar_value=grad_norm, global_step=global_step)

                lr_scheduler.step()
                optimizer.zero_grad()

        else:
            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch}",
                disable=not is_master(),
                initial=start_step // args.accumulation_steps,
            )
            total_loss = torch.tensor(0.0, device=get_current_device())
            for step, batch in enumerate(pbar, start=start_step // args.accumulation_steps):
                batch = {k: v.to(get_current_device()) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                batch_output = model(**batch)

                loss = batch_output.loss / args.accumulation_steps
                total_loss.add_(loss.data)

                booster.backward(loss=loss, optimizer=optimizer)

                if (step + 1) % args.accumulation_steps == 0:
                    all_reduce_mean(total_loss, plugin)

                    optimizer.step()

                    grad_norm = optimizer.get_grad_norm()
                    pbar.set_postfix({"loss": total_loss.item(), "grad_norm": grad_norm})
                    if args.tensorboard_dir is not None and is_master():
                        global_step = (epoch * num_steps_per_epoch) + (step + 1) // args.accumulation_steps
                        writer.add_scalar(tag="Loss", scalar_value=total_loss.item(), global_step=global_step)
                        writer.add_scalar(
                            tag="Learning Rate",
                            scalar_value=lr_scheduler.get_last_lr()[0],
                            global_step=global_step,
                        )
                        writer.add_scalar(tag="Grad Norm", scalar_value=grad_norm, global_step=global_step)

                    lr_scheduler.step()
                    optimizer.zero_grad()

                    total_loss.fill_(0.0)

        # Delete cache.
        # del batch, batch_labels, batch_output, loss
        accelerator.empty_cache()

    # Final save.
    coordinator.print_on_master("Start saving final model checkpoint")
    if args.lora_rank > 0:
        booster.save_lora_as_pretrained(model, os.path.join(args.save_dir, "lora"))
    else:
        booster.save_model(model, os.path.join(args.save_dir, "modeling"), shard=True)
    coordinator.print_on_master(f"Saved final model checkpoint at epoch {epoch} at folder {args.save_dir}")

    coordinator.print_on_master(f"Max device memory usage: {accelerator.max_memory_allocated()/1024**2:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Basic training information.
    parser.add_argument(
        "-m",
        "--pretrained",
        type=str,
        required=True,
        help="Address of the pre-trained model",
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Raw Jonl dataset for training.")
    parser.add_argument(
        "-p",
        "--plugin",
        type=str,
        default="zero2",
        choices=["gemini", "gemini_auto", "zero2", "zero2_cpu", "3d", "ddp", "moe"],
        help="Choose which plugin to use",
    )
    parser.add_argument("--save_dir", type=str, default="checkpoint_dir", help="Checkpoint directory")
    parser.add_argument("--tensorboard_dir", type=str, default=None, help="Tensorboard directory")
    parser.add_argument("--config_file", type=str, default="training_config.json", help="Config file")
    # Training parameters
    parser.add_argument("-n", "--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of accumulation steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Global Batch size of each process")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=8192, help="Model max length")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16"],
        help="Mixed precision",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    parser.add_argument(
        "-g",
        "--use_grad_checkpoint",
        action="store_true",
        default=False,
        help="Use gradient checkpointing",
    )
    parser.add_argument(
        "-f",
        "--use_flash_attn",
        action="store_true",
        default=False,
        help="Use flash-attention",
    )

    # Additional arguments for 3d plugin.
    parser.add_argument("--tp", type=int, default=1, help="TP size, used for 3d plugin.")
    parser.add_argument("--pp", type=int, default=1, help="PP size, used for 3d plugin.")
    parser.add_argument("--sp", type=int, default=1, help="SP size, used for 3d plugin.")
    parser.add_argument("--ep", type=int, default=1, help="EP size, used for moe plugin.")
    parser.add_argument("--zero_stage", type=int, default=1, help="Zero stage, used for 3d plugin.", choices=[0, 1, 2])
    parser.add_argument(
        "--sp_mode",
        type=str,
        default="split_gather",
        choices=["split_gather", "ring", "all_to_all"],
        help="SP mode, used for 3d plugin.",
    )
    parser.add_argument(
        "--enable_sequence_parallelism",
        default=False,
        action="store_true",
        help="Whether to enable SP, used for 3d plugin.",
    )
    parser.add_argument(
        "--zero_cpu_offload", default=False, action="store_true", help="Whether to use offloading, used for 3d plugin."
    )
    parser.add_argument(
        "--microbatch_size", type=int, default=1, help="Batch size for each process in PP, used for 3d plugin."
    )
    parser.add_argument("--lora_rank", type=int, default=0, help="lora rank when using lora to train.")
    parser.add_argument("--lora_alpha", type=int, default=8, help="lora alpha when using lora to train.")

    args = parser.parse_args()

    if args.plugin in ["3d", "moe"] and args.pp > 1 and args.accumulation_steps > 1:
        raise ValueError("Accumulation steps should be 1 when using PP. Please adjust batch size directly.")

    train(args)
