import argparse

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.cluster import DistCoordinator


def parse_args():
    # basic settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--plugin",
        type=str,
        default="ep",
        choices=["ep"],
        help="Parallel methos.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="The mixed precision training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Launch ColossalAI
    colossalai.launch_from_torch(seed=args.seed)
    coordinator = DistCoordinator()

    config = MixtralConfig.from_pretrained(args.model_name)
    ep_size = min(dist.get_world_size(), config.num_local_experts)
    # Set plugin
    if args.plugin == "ep":
        plugin = MoeHybridParallelPlugin(
            tp_size=1,
            pp_size=1,
            ep_size=ep_size,
            zero_stage=1,
            precision=args.precision,
            enable_fused_normalization=args.use_layernorm_kernel,
            enable_jit_fused=args.use_kernel,
        )
    else:
        raise ValueError(f"Invalid plugin {args.plugin}")
    coordinator.print_on_master(f"Set plugin as {plugin.__class__.__name__}")

    # Build mixtral model
    model = MixtralForCausalLM.from_pretrained(args.model_name)
    coordinator.print_on_master(f"Finish load model")

    # Prepare tokenizer and dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set booster
    booster = Booster(plugin=plugin)
    model, _, _, _, _ = booster.boost(model=model)
    coordinator.print_on_master(f"Finish init booster")

    model.eval()

    if coordinator.rank == 0:
        text = ["Hello my name is"]
    else:
        text = [
            "What's the largest country in the world?",
            "How many people live in China?",
            "帮我续写这首诗：离离原上草",
        ]
    tokenizer.pad_token = tokenizer.unk_token
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(torch.cuda.current_device())

    with torch.no_grad():
        outputs = model.module.generate(**inputs, max_new_tokens=20)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"[{coordinator.rank}] {outputs}")


if __name__ == "__main__":
    main()
