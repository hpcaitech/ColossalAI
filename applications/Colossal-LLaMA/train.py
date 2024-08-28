#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continual Pre-training/Supervised fine-tuning of Colossal-LLaMA-2 developed by Colossal-AI Team
"""

import argparse
import json
import os
import resource
from contextlib import nullcontext

import torch
from colossal_llama.dataset.dummy_dataset import RandomDataset
from colossal_llama.dataset.loader import (
    DataCollatorForSupervisedDataset,
    StatefulDistributedSampler,
    load_tokenized_dataset,
)
from colossal_llama.utils.ckpt_io import load_checkpoint, save_checkpoint
from colossal_llama.utils.froze import freeze_non_embeds_parameters
from colossal_llama.utils.neftune_patch import activate_neftune, deactivate_neftune
from colossal_llama.utils.utils import all_reduce_mean, format_numel_str, get_model_numel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device


def train(args) -> None:
    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch()
    accelerator = get_accelerator()
    coordinator = DistCoordinator()

    # ==============================
    # Initialize Tensorboard and Save Config
    # ==============================
    if coordinator.is_master():
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)

        with open(args.config_file, "w") as f:
            json.dump(args.__dict__, f, indent=4)

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
            enable_fused_normalization=torch.cuda.is_available(),
            enable_flash_attention=args.use_flash_attn,
        )
    elif args.plugin == "gemini_auto":
        plugin = GeminiPlugin(
            precision=args.mixed_precision,
            placement_policy="auto",
            initial_scale=2**16,
            max_norm=args.grad_clip,
            enable_gradient_accumulation=(args.accumulation_steps > 1),
            enable_fused_normalization=torch.cuda.is_available(),
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
            enable_fused_normalization=torch.cuda.is_available(),
            enable_sequence_parallelism=args.enable_sequence_parallelism,
            cpu_offload=True if args.zero_stage >= 1 and args.zero_cpu_offload else False,
            parallel_output=False,
            max_norm=args.grad_clip,
            precision=args.mixed_precision,
            microbatch_size=args.microbatch_size,
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    booster = Booster(plugin=plugin)

    # ======================================================
    # Initialize Tokenizer, Dataset, Collator and Dataloader
    # ======================================================
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    if args.pad_token == "eos":
        tokenizer.pad_token = tokenizer.eos_token
    elif args.pad_token == "unk":
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    coordinator.print_on_master(
        f"Training Info:\nConfig file: {args.config_file} \nTensorboard logs: {args.tensorboard_dir} \nModel checkpoint: {args.save_dir}"
    )

    if args.benchmark:
        coordinator.print_on_master(f"Run benchmark with {args.num_samples} random samples.")
        dataset = RandomDataset(
            num_samples=args.num_samples, max_length=args.max_length, vocab_size=tokenizer.vocab_size
        )
        dataloader = plugin.prepare_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            seed=42,
            distributed_sampler_cls=StatefulDistributedSampler,
        )
    else:
        coordinator.print_on_master(f"Load dataset: {args.dataset}")
        dataset = load_tokenized_dataset(dataset_paths=args.dataset, mode="train")
        data_collator = DataCollatorForSupervisedDataset(
            tokenizer=tokenizer, max_length=args.max_length, padding=args.padding_mode
        )
        dataloader = plugin.prepare_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=data_collator,
            distributed_sampler_cls=StatefulDistributedSampler,
        )

    coordinator.print_on_master(
        f"Max device memory after data loader: {accelerator.max_memory_allocated() / 1024 ** 2:.2f} MB"
    )

    # ======================================================
    # Initialize Model, Objective, Optimizer and LR Scheduler
    # ======================================================
    init_ctx = (
        LazyInitContext(default_device=get_current_device())
        if isinstance(plugin, (GeminiPlugin, HybridParallelPlugin))
        else nullcontext()
    )
    with init_ctx:
        if args.use_flash_attn:
            model = AutoModelForCausalLM.from_pretrained(
                args.pretrained,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.pretrained,
                torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
                trust_remote_code=True,
            )
        # Freeze part of parameters.
        if args.freeze_non_embeds_params:
            freeze_non_embeds_parameters(model=model)
    # this is essential, otherwise the grad checkpoint will not work.
    model.train()

    if args.use_grad_checkpoint:
        model.gradient_checkpointing_enable()
        coordinator.print_on_master(msg="Gradient checkpointing enabled successfully")

    model_numel = get_model_numel(model)
    coordinator.print_on_master(f"Model params: {format_numel_str(model_numel)}")

    optimizer = HybridAdam(
        model_params=(
            filter(lambda p: p.requires_grad, model.parameters())
            if args.freeze_non_embeds_params
            else model.parameters()
        ),
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

    coordinator.print_on_master(
        f"Booster init max device memory: {accelerator.max_memory_allocated() / 1024 ** 2:.2f} MB"
    )
    coordinator.print_on_master(
        f"Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB"
    )

    start_epoch = 0
    start_step = 0
    sampler_start_idx = 0
    if args.load_checkpoint is not None:
        if "modeling" in args.load_checkpoint:
            coordinator.print_on_master(f"Continued pretrain from checkpoint {args.load_checkpoint}")
            booster.load_model(model, args.load_checkpoint)
        else:
            coordinator.print_on_master(f"Load model checkpoint from {args.load_checkpoint}")
            start_epoch, start_step, sampler_start_idx = load_checkpoint(
                load_dir=args.load_checkpoint,
                booster=booster,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
            coordinator.print_on_master(
                f"Loaded checkpoint {args.load_checkpoint} at epoch {start_epoch} step {start_step}"
            )
            coordinator.print_on_master(f"Loaded sample at index {sampler_start_idx}")

        coordinator.print_on_master(
            f"Checkpoint loaded max device memory: {accelerator.max_memory_allocated() / 1024 ** 2:.2f} MB"
        )
        coordinator.print_on_master(
            f"Checkpoint loaded device memory: {accelerator.memory_allocated() / 1024 ** 2:.2f} MB"
        )
        coordinator.print_on_master(
            f"Checkpoint loaded max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB"
        )

    if args.use_neft:
        coordinator.print_on_master("Activate NEFTune.")
        model, handle = activate_neftune(model)

    num_steps_per_epoch = len(dataloader) // args.accumulation_steps
    # If resume training, set the sampler start index to the correct value
    assert isinstance(dataloader.sampler, StatefulDistributedSampler)
    dataloader.sampler.set_start_index(start_index=sampler_start_idx)

    for epoch in range(start_epoch, args.num_epochs):
        dataloader.sampler.set_epoch(epoch=epoch)
        if isinstance(plugin, HybridParallelPlugin) and plugin.pp_size > 1:
            data_iter = iter(dataloader)
            step_bar = tqdm(
                range(len(dataloader)),
                desc="Step",
                disable=not (coordinator._local_rank == coordinator._world_size - 1),
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
                    if coordinator._local_rank == coordinator._world_size - 1:
                        step_bar.set_postfix({"train/loss": global_loss.item()})
                optimizer.step()
                optimizer.zero_grad()

                # Save modeling.
                save_model_condition = args.save_interval > 0 and (step + 1) % args.save_interval == 0

                if not args.skip_save_each_epoch:
                    save_model_condition = save_model_condition or (step + 1) == len(dataloader)

                if save_model_condition and not args.benchmark:
                    coordinator.print_on_master("\nStart saving model checkpoint with running states")

                    if args.use_neft:
                        coordinator.print_on_master("Deactivate NEFTune before saving model.")
                        deactivate_neftune(model, handle)

                    accelerator.empty_cache()
                    save_checkpoint(
                        save_dir=args.save_dir,
                        booster=booster,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        epoch=epoch,
                        step=step + 1,
                        batch_size=args.batch_size,
                        coordinator=coordinator,
                    )
                    coordinator.print_on_master(
                        f"Saved checkpoint at epoch {epoch} step {step + 1} at folder {args.save_dir}"
                    )

                    if args.use_neft:
                        coordinator.print_on_master("Activate NEFTune.")
                        model, handle = activate_neftune(model)
        else:
            pbar = tqdm(
                desc=f"Epoch {epoch}",
                disable=not coordinator.is_master(),
                total=num_steps_per_epoch,
                initial=start_step // args.accumulation_steps,
            )
            total_loss = torch.tensor(0.0, device=get_current_device())
            for step, batch in enumerate(dataloader, start=start_step):
                batch = {k: v.to(get_current_device()) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                batch_output = model(**batch)

                loss = batch_output.loss / args.accumulation_steps
                total_loss.add_(loss.data)

                booster.backward(loss=loss, optimizer=optimizer)

                if (step + 1) % args.accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    all_reduce_mean(tensor=total_loss)
                    pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})
                    if coordinator.is_master():
                        global_step = (epoch * num_steps_per_epoch) + (step + 1) // args.accumulation_steps
                        writer.add_scalar(tag="Loss", scalar_value=total_loss.item(), global_step=global_step)
                        writer.add_scalar(
                            tag="Learning Rate",
                            scalar_value=lr_scheduler.get_last_lr()[0],
                            global_step=global_step,
                        )
                    total_loss.fill_(0.0)
                    pbar.update()

            # Save modeling.
            save_model_condition = (
                args.save_interval > 0 and (step + 1) % (args.save_interval * args.accumulation_steps) == 0
            )

            if not args.skip_save_each_epoch:
                save_model_condition = save_model_condition or (step + 1) == len(dataloader)

            if save_model_condition and not args.benchmark:
                coordinator.print_on_master("\nStart saving model checkpoint with running states")

                if args.use_neft:
                    coordinator.print_on_master("Deactivate NEFTune before saving model.")
                    deactivate_neftune(model, handle)

                accelerator.empty_cache()
                save_checkpoint(
                    save_dir=args.save_dir,
                    booster=booster,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    epoch=epoch,
                    step=step + 1,
                    batch_size=args.batch_size,
                    coordinator=coordinator,
                )
                coordinator.print_on_master(
                    f"Saved checkpoint at epoch {epoch} step {step + 1} at folder {args.save_dir}"
                )

                if args.use_neft:
                    coordinator.print_on_master("Activate NEFTune.")
                    model, handle = activate_neftune(model)

            # Delete cache.
            # del batch, batch_labels, batch_output, loss
            accelerator.empty_cache()

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(start_index=0)
        start_step = 0

    if args.use_neft:
        coordinator.print_on_master("Deactivate NEFTune.")
        deactivate_neftune(model, handle)

    # Final save.
    if not args.benchmark:
        coordinator.print_on_master("Start saving final model checkpoint")
        booster.save_model(model, os.path.join(args.save_dir, "modeling"), shard=True)
        coordinator.print_on_master(f"Saved final model checkpoint at epoch {epoch} at folder {args.save_dir}")

    coordinator.print_on_master(f"Max device memory usage: {accelerator.max_memory_allocated()/1024**2:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Basic training information.
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Address of the pre-trained model",
    )
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Load checkpoint for continuous training.")
    parser.add_argument("--dataset", nargs="+", default=[])
    parser.add_argument(
        "--plugin",
        type=str,
        default="gemini",
        choices=["gemini", "gemini_auto", "zero2", "zero2_cpu", "3d", "ddp"],
        help="Choose which plugin to use",
    )
    parser.add_argument("--save_interval", type=int, default=1000, help="Save interval")
    parser.add_argument("--save_dir", type=str, default="checkpoint_dir", help="Checkpoint directory")
    parser.add_argument("--tensorboard_dir", type=str, default="logs_dir", help="Tensorboard directory")
    parser.add_argument("--config_file", type=str, default="config_file", help="Config file")
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of accumulation steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Global Batch size of each process")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=8192, help="Model max length")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Mixed precision",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    parser.add_argument(
        "--use_grad_checkpoint",
        action="store_true",
        default=False,
        help="Use gradient checkpointing",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        default=False,
        help="Use flash-attention",
    )
    parser.add_argument(
        "--use_neft",
        action="store_true",
        default=False,
        help="Use NEFTune",
    )
    parser.add_argument(
        "--freeze_non_embeds_params",
        action="store_true",
        default=False,
        help="Freeze non embeddings parameters",
    )
    parser.add_argument("--pad_token", choices=["eos", "unk"], default="eos")
    parser.add_argument("--padding_mode", choices=["max_length", "longest"], default="max_length")
    parser.add_argument(
        "--skip_save_each_epoch",
        action="store_true",
        default=False,
        help="Skip saving the model checkpoint after each epoch is completed.",
    )

    # Additional arguments for 3d plugin.
    parser.add_argument("--tp", type=int, default=1, help="TP size, used for 3d plugin.")
    parser.add_argument("--pp", type=int, default=1, help="PP size, used for 3d plugin.")
    parser.add_argument("--sp", type=int, default=1, help="SP size, used for 3d plugin.")
    parser.add_argument("--zero_stage", type=int, default=0, help="Zero stage, used for 3d plugin.", choices=[0, 1, 2])
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

    # Additional arguments for benchmark.
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples for benchmarking.")
    parser.add_argument(
        "--benchmark", action="store_true", default=False, help="Benchmark performance using random dataset."
    )
    args = parser.parse_args()
    train(args)
