import argparse
import json
import os

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from model.modeling_openmoe import OpenMoeForCausalLM, set_openmoe_args
from model.openmoe_policy import OpenMoeForCausalLMPolicy
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import T5Tokenizer
from transformers.models.llama import LlamaConfig
from utils import PerformanceEvaluator, get_model_numel

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.legacy.moe.manager import MOE_MANAGER
from colossalai.legacy.moe.utils import skip_init
from colossalai.moe.layers import apply_load_balance
from colossalai.nn.optimizer import HybridAdam


def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def load_ckpt(repo_name: str, model: OpenMoeForCausalLM, booster: Booster):
    ckpt_path = snapshot_download(repo_name)
    # single ckpt
    if os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin")):
        ckpt_path = os.path.join(ckpt_path, "pytorch_model.bin")
    # shard ckpt
    elif os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin.index.json")):
        ckpt_path = os.path.join(ckpt_path, "pytorch_model.bin.index.json")
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")
    booster.load_model(model, ckpt_path)


class RandomDataset(Dataset):
    def __init__(
        self, num_samples: int = 1000, max_length: int = 2048, vocab_size: int = 256384, tokenizer: T5Tokenizer = None
    ):
        self.num_samples = num_samples
        self.max_length = max_length
        if os.path.exists("./mock_data.json"):
            self.input_ids = []
            self.attention_mask = []
            with open("./mock_data.json", "r") as f:
                data = json.load(f)
            for v in data.values():
                d = v["text"]
                encode = tokenizer(
                    "<pad>" + d,
                    return_tensors="pt",
                    add_special_tokens=False,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )
                self.input_ids.append(encode["input_ids"])
                self.attention_mask.append(encode["attention_mask"])
            self.input_ids = torch.cat(self.input_ids, dim=0).to(get_accelerator().get_current_device())
            self.attention_mask = torch.cat(self.attention_mask, dim=0).to(get_accelerator().get_current_device())
            repeat_times = num_samples // self.input_ids.shape[0] + 1
            self.input_ids = self.input_ids.repeat(repeat_times, 1)[:num_samples]
            self.attention_mask = self.attention_mask.repeat(repeat_times, 1)[:num_samples]
        else:
            self.input_ids = torch.randint(
                0, vocab_size, (num_samples, max_length), device=get_accelerator().get_current_device()
            )
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
        default="base",
        choices=["base", "8b"],
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (per dp group) for the training dataloader.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
        help="sequence length for the training dataloader.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--plugin",
        type=str,
        default="hybrid",
        help="parallel plugin",
    )
    # hybrid plugin
    parser.add_argument("--pp_size", type=int, default=2, help="pp size")
    parser.add_argument("--dp_size", type=int, default=1, help="dp size")
    parser.add_argument("--ep_size", type=int, default=2, help="ep size")
    parser.add_argument("--zero_stage", type=int, default=2, help="zero stage in hybrid plugin")
    parser.add_argument("--microbatch_size", type=int, default=1, help="microbatch size")
    parser.add_argument("--extra_dp_size", type=int, default=1)
    # kernel
    parser.add_argument(
        "--use_kernel",
        action="store_true",
        help="Use kernel optim. Need to install flash attention, apex, triton to enable all kernel optimizations.",
    )
    # bench
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--active", type=int, default=20)
    # load balance
    parser.add_argument("--load_balance", action="store_true")

    # overlap communication
    parser.add_argument("--overlap_comm", action="store_true")
    # hierarchical all-to-all
    parser.add_argument("--hierarchical_alltoall", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Launch ColossalAI
    colossalai.launch_from_torch(seed=args.seed)
    coordinator = DistCoordinator()

    # Set plugin
    booster_kwargs = {}
    hybrid_dict = {
        "tp_size": 1,
        "custom_policy": OpenMoeForCausalLMPolicy(),
        "enable_fused_normalization": args.use_kernel,
        "enable_jit_fused": args.use_kernel,
        "precision": "bf16",
        "zero_stage": args.zero_stage,
    }
    mgr_dict = {}
    if args.plugin == "ep":
        dp_size = dist.get_world_size()
        plugin = MoeHybridParallelPlugin(
            pp_size=1,
            **hybrid_dict,
        )
        MOE_MANAGER.setup(
            parallel="EP",
            max_ep_size=dp_size,
            **mgr_dict,
        )
    elif args.plugin == "ep_zero":
        dp_size = dist.get_world_size()
        use_ep_inside = False
        plugin = MoeHybridParallelPlugin(
            pp_size=1,
            ep_size=args.ep_size,
            use_ep_inside=use_ep_inside,
            **hybrid_dict,
        )
        MOE_MANAGER.setup(
            parallel="EP",
            max_ep_size=dp_size // args.extra_dp_size,
            use_ep_inside=use_ep_inside,
            **mgr_dict,
        )
    elif args.plugin == "hybrid":
        dp_size = dist.get_world_size() // args.pp_size
        plugin = MoeHybridParallelPlugin(
            pp_size=args.pp_size,
            zero_stage=args.zero_stage,
            microbatch_size=args.microbatch_size,
            **hybrid_dict,
        )
        MOE_MANAGER.setup(
            parallel="EP",
            mode="fixed",
            fixed_dp_size=args.dp_size,
            fixed_ep_size=args.ep_size,
            fixed_pp_size=args.pp_size,
            **mgr_dict,
        )
    else:
        raise ValueError(f"Invalid plugin {args.plugin}")
    coordinator.print_on_master(f"Set plugin as {plugin}")

    # Build OpenMoe model
    repo_name = "hpcai-tech/openmoe-" + args.model_name
    config = LlamaConfig.from_pretrained(repo_name)
    set_openmoe_args(
        config,
        num_experts=config.num_experts,
        moe_layer_interval=config.moe_layer_interval,
        enable_load_balance=args.load_balance,
        enable_kernel=args.use_kernel,
        enable_comm_overlap=args.overlap_comm,
        enable_hierarchical_alltoall=args.hierarchical_alltoall,
    )
    with skip_init():
        model = OpenMoeForCausalLM(config)
    coordinator.print_on_master(f"Finish init model with config:\n{config}")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Prepare tokenizer and dataloader
    tokenizer = T5Tokenizer.from_pretrained("google/umt5-small")
    dataset = RandomDataset(
        num_samples=args.batch_size * (args.warmup + args.active + 1) * dp_size,
        max_length=args.seq_length,
        tokenizer=tokenizer,
    )
    dataloader = plugin.prepare_dataloader(dataset, batch_size=args.batch_size)

    # Set optimizer
    optimizer = HybridAdam(model.parameters(), weight_decay=0.01, lr=1e-5)

    model_numel = get_model_numel(model)
    performance_evaluator = PerformanceEvaluator(
        model_numel,
        enable_grad_checkpoint=True,
        ignore_steps=args.warmup,
        dp_world_size=dp_size,
    )

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    load_ckpt(repo_name, model, booster)
    model, optimizer, _, dataloader, _ = booster.boost(model=model, optimizer=optimizer, dataloader=dataloader)
    use_pipeline = isinstance(booster.plugin, MoeHybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    coordinator.print_on_master(f"Finish init booster")

    # Start finetuning
    coordinator.print_on_master(f"Start training")
    model.train()
    train_dataloader_iter = iter(dataloader)
    total_len = len(train_dataloader_iter) - 1
    exmaple_data = next(train_dataloader_iter)
    with tqdm(range(total_len), disable=not coordinator.is_master()) as pbar:
        for step in pbar:
            performance_evaluator.on_step_start(step)
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
                    pbar.set_postfix({"loss": loss.item()})
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
            optimizer.zero_grad()
            performance_evaluator.on_step_end(exmaple_data["input_ids"])
            if (step == args.warmup // 2) and args.load_balance:
                coordinator.print_on_master(f"Apply load balance")
                apply_load_balance(model, optimizer)
    performance_evaluator.on_fit_end()


if __name__ == "__main__":
    main()
