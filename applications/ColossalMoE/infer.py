import argparse

import torch
import torch.distributed as dist
from colossal_moe.models.mixtral_checkpoint import MixtralMoECheckpointIO
from colossal_moe.models.mixtral_layer import replace_moe_layer
from colossal_moe.models.mixtral_policy import MixtralForCausalLMPolicy
from colossal_moe.utils import load_model
from transformers import AutoTokenizer
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.moe import MOE_MANAGER
from colossalai.moe.utils import skip_init
from colossalai.utils import get_current_device


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
        default="hybrid",
        choices=["ep"],
        help="Parallel methos.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs",
        help="The path of your saved model after finetuning.",
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
    colossalai.launch_from_torch(config={}, seed=args.seed)
    coordinator = DistCoordinator()

    # Set plugin
    booster_kwargs = {}
    hybrid_dict = {
        "tp_size": 1,
        "custom_policy": MixtralForCausalLMPolicy(),
        "enable_fused_normalization": args.use_layernorm_kernel,
        "enable_jit_fused": args.use_kernel,
        "precision": args.precision,
        "checkpoint_io": MixtralMoECheckpointIO,
        "zero_stage": 1,
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
    else:
        raise ValueError(f"Invalid plugin {args.plugin}")
    coordinator.print_on_master(f"Set plugin as {plugin.__class__.__name__}")

    # Build mixtral model
    config = MixtralConfig.from_pretrained(args.model_name)
    config.num_local_experts = 1  # dont change this. it will not affect model
    with skip_init():
        model = MixtralForCausalLM(config)
    model.num_experts = 8
    model = model.to(torch.bfloat16) if args.precision == "bf16" else model.to(torch.float16)
    model = model.to(get_current_device())
    coordinator.print_on_master(f"Finish init model with config:\n{config}")

    # Replace moe
    with skip_init():
        replace_moe_layer(model)
    model.eval()
    coordinator.print_on_master(f"Finish replace moe module")

    # Prepare tokenizer and dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, _, _, _, _ = booster.boost(model=model)
    coordinator.print_on_master(f"Finish init booster")

    # load ckpt
    load_model(args.model_name, model, booster)
    coordinator.print_on_master(f"Finish load ckpt")

    text = ["Hello my name is", "1+1=?"]
    tokenizer.pad_token = tokenizer.unk_token
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(torch.cuda.current_device())
    outputs = model.module.generate(**inputs, max_new_tokens=20)
    outputs = tokenizer.batch_decode(outputs)[0]
    print(outputs)


if __name__ == "__main__":
    main()
