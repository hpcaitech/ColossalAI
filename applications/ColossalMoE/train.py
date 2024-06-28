import argparse

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.mixtral import MixtralForCausalLM
from utils import load_checkpoint, move_to_cuda, save_checkpoint

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device


@torch.no_grad()
def get_global_loss(loss, booster):
    global_loss = loss.clone().detach()
    dist.all_reduce(tensor=global_loss, op=dist.ReduceOp.SUM, group=booster.plugin.dp_group)
    global_loss.div_(booster.plugin.dp_size)
    return global_loss


class RandomDataset(Dataset):
    def __init__(self, num_samples: int = 1000, max_length: int = 2048, vocab_size: int = 100, tokenizer=None):
        self.num_samples = num_samples
        self.max_length = max_length
        self.input_ids = torch.randint(0, vocab_size, (num_samples, max_length), device=get_current_device())
        self.attention_mask = torch.ones_like(self.input_ids)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],
        }


def parse_args():
    # basic settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Load checkpoint")
    parser.add_argument(
        "--plugin",
        type=str,
        default="hybrid",
        choices=["hybrid"],
        help="Parallel methods.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs",
        help="The path of your saved model after finetuning.",
    )
    parser.add_argument("--num_epoch", type=int, default=1, help="Number of epochs.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per dp group) for the training dataloader.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help=" The interval (steps) of saving checkpoints.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="The mixed precision training.",
    )
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # optim
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    # lr scheduler
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")

    # zero stage for all plugins
    parser.add_argument("--zero_stage", type=int, default=2, help="zero stage.")
    # hybrid plugin
    parser.add_argument("--pp_size", type=int, default=2, help="pp size for hybrid plugin")
    parser.add_argument("--dp_size", type=int, default=1, help="dp size for hybrid plugin")
    parser.add_argument("--ep_size", type=int, default=2, help="ep size for hybrid plugin")
    parser.add_argument("--microbatch_size", type=int, default=1, help="Microbatch size in pipeline for hybrid plugin")

    # kernel
    parser.add_argument(
        "--use_kernel",
        action="store_true",
        help="Use kernel optim. Need to install flash attention and triton to enable all kernel optimizations. Skip if not installed.",
    )
    parser.add_argument(
        "--use_layernorm_kernel",
        action="store_true",
        help="Use layernorm kernel. Need to install apex. Raise error if not installed.",
    )

    # load balance
    parser.add_argument(
        "--load_balance", action="store_true", help="Expert load balance. Defaults to False. Recommend to enable."
    )
    parser.add_argument("--load_balance_interval", type=int, default=1000, help="Expert load balance interval.")
    # communicate overlap
    parser.add_argument(
        "--comm_overlap",
        action="store_true",
        help="Use communication overlap for MoE. Recommended to enable for multi-node training.",
    )
    # hierarchical all-to-all
    parser.add_argument(
        "--hierarchical_alltoall",
        action="store_true",
        help="Use hierarchical all-to-all for MoE. Recommended to enable for multi-node training.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Launch ColossalAI
    colossalai.launch_from_torch(seed=args.seed)
    coordinator = DistCoordinator()

    # Set plugin
    if args.plugin == "hybrid":
        plugin = MoeHybridParallelPlugin(
            tp_size=1,
            pp_size=args.pp_size,
            ep_size=args.ep_size,
            microbatch_size=args.microbatch_size,
            enable_fused_normalization=args.use_layernorm_kernel,
            enable_jit_fused=args.use_kernel,
            precision=args.precision,
            zero_stage=args.zero_stage,
        )

    else:
        raise ValueError(f"Invalid plugin {args.plugin}")
    coordinator.print_on_master(f"Set plugin as {plugin.__class__.__name__}")

    # Build Mixtral model
    model = MixtralForCausalLM.from_pretrained(args.model_name)
    coordinator.print_on_master(f"Finish init model")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Prepare tokenizer and dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = RandomDataset(num_samples=100, tokenizer=tokenizer)
    collate_fn = None
    dataloader = plugin.prepare_dataloader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn
    )

    # Set optimizer
    optimizer = HybridAdam(
        model_params=model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        adamw_mode=True,
    )

    # Set lr scheduler
    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer,
        total_steps=args.num_epochs * len(dataloader),
        warmup_steps=(
            args.warmup_steps if args.warmup_steps is not None else int(args.num_epochs * len(dataloader) * 0.025)
        ),
        eta_min=0.1 * args.lr,
    )

    # Set booster
    booster = Booster(plugin=plugin)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    use_pipeline = isinstance(booster.plugin, MoeHybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    coordinator.print_on_master(f"Finish init booster")

    # Load ckpt
    if args.load_checkpoint is not None:
        load_checkpoint(args.load_checkpoint, booster, model, optimizer, lr_scheduler)
        coordinator.print_on_master(f"Finish load optimizer")

    # Start finetuning
    coordinator.print_on_master(f"Start finetuning")
    for epoch in range(args.num_epoch):
        model.train()
        train_dataloader_iter = iter(dataloader)
        total_len = len(train_dataloader_iter)
        with tqdm(
            range(total_len),
            desc=f"Epoch [{epoch + 1}/{args.num_epoch}]",
            disable=not coordinator.is_master() if use_pipeline == False else not is_pp_last_stage,
        ) as pbar:
            for step in pbar:
                if use_pipeline:
                    # Forward pass
                    outputs = booster.execute_pipeline(
                        train_dataloader_iter,
                        model,
                        lambda x, y: x.loss,
                        optimizer,
                        return_loss=True,
                    )
                    # Backward and optimize
                    if is_pp_last_stage:
                        loss = outputs["loss"]
                        global_loss = get_global_loss(loss, booster)
                        if coordinator._local_rank == "0":
                            pbar.set_postfix({"Loss": global_loss.item()})
                else:
                    # Forward pass
                    data = next(train_dataloader_iter)
                    data = move_to_cuda(data, torch.cuda.current_device())
                    outputs = model(**data)
                    loss = outputs["loss"]
                    # Backward
                    booster.backward(loss, optimizer)
                    pbar.set_postfix({"loss": loss.item()})

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Apply load balance
                # if (
                #     args.load_balance
                #     and args.load_balance_interval > 0
                #     and (step + 1) % args.load_balance_interval == 0
                # ):
                #     coordinator.print_on_master(f"Apply load balance")
                #     apply_load_balance(model, optimizer)
                # save checkpoint
                if (step + 1) % args.save_interval == 0:
                    coordinator.print_on_master(f"Saving model checkpoint to {args.output_path}")
                    save_checkpoint(
                        args.output_path,
                        booster,
                        model,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step,
                        args.batch_size,
                        coordinator,
                    )

        # save checkpoint at the end of each epochs
        booster.save_model(model, args.output_path, shard=True, size_per_shard=5120)
        coordinator.print_on_master(f"Saving model checkpoint to {args.output_path}")

    # Finish training
    coordinator.print_on_master(f"Finish training")


if __name__ == "__main__":
    main()
