import datasets
import torch
import torch.distributed as dist
import transformers
from model.modeling_openmoe import OpenMoeForCausalLM
from model.openmoe_policy import OpenMoeForCausalLMPolicy
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Adafactor
from transformers.models.llama import LlamaConfig
from utils import PerformanceEvaluator, get_model_numel

import colossalai
from colossalai import get_default_parser
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.moe.manager import MOE_MANAGER
from colossalai.utils import get_current_device


def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


class RandomDataset(Dataset):

    def __init__(self, num_samples: int = 1000, max_length: int = 2048, vocab_size: int = 32000):
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
    parser = get_default_parser()
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
        choices=["zero1", "zero2", "hybrid"],
    )
    # hybrid plugin
    parser.add_argument("--pp_size", type=int, default=2, help="pp size")
    parser.add_argument("--dp_size", type=int, default=1, help="dp size")
    parser.add_argument("--ep_size", type=int, default=2, help="ep size")
    parser.add_argument("--zero_stage", type=int, default=1, help="zero stage in hybrid plugin")
    parser.add_argument("--microbatch_size", type=int, default=1, help="microbatch size")
    # kernel
    parser.add_argument(
        "--use_kernel",
        action="store_true",
        help="Use kernel optim. Need to install flash attention, apex, triton to enable all kernel optimizations.",
    )
    # bench
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--active", type=int, default=20)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Launch ColossalAI
    colossalai.launch_from_torch(config={}, seed=args.seed)
    coordinator = DistCoordinator()

    # Manage loggers
    disable_existing_loggers()
    logger = get_dist_logger()
    if coordinator.is_master():
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set plugin
    booster_kwargs = {}
    if args.plugin == "zero1":
        dp_size = dist.get_world_size()
        plugin = LowLevelZeroPlugin(initial_scale=2**5, stage=1)
        MOE_MANAGER.setup(
            seed=42,
            parallel="EP",
            use_kernel_optim=args.use_kernel,
        )
    elif args.plugin == "zero2":
        dp_size = dist.get_world_size()
        plugin = LowLevelZeroPlugin(initial_scale=2**5, stage=2)
        MOE_MANAGER.setup(
            seed=42,
            parallel="EP",
            use_kernel_optim=args.use_kernel,
        )
    elif args.plugin == "hybrid":
        dp_size = dist.get_world_size() // args.pp_size
        plugin = MoeHybridParallelPlugin(
            tp_size=1,
            pp_size=args.pp_size,
            zero_stage=args.zero_stage,
            microbatch_size=args.microbatch_size,
            custom_policy=OpenMoeForCausalLMPolicy(),
            enable_fused_normalization=args.use_kernel,
            enable_jit_fused=args.use_kernel,
        )
        MOE_MANAGER.setup(
            seed=42,
            parallel="EP",
            mode="fixed",
            fixed_dp_size=args.dp_size,
            fixed_ep_size=args.ep_size,
            fixed_pp_size=args.pp_size,
            use_kernel_optim=args.use_kernel,
        )
    else:
        raise ValueError(f"Invalid plugin {args.plugin}")
    logger.info(f"Set plugin as {plugin}", ranks=[0])

    # Build OpenMoe model
    repo_name = "hpcaitech/openmoe-" + args.model_name
    config = LlamaConfig.from_pretrained(repo_name)
    setattr(config, "router_aux_loss_factor", 0.1)
    setattr(config, "router_z_loss_factor", 0.1)
    setattr(config, "label_smoothing", 0.1)
    setattr(config, "z_loss_factor", 0.1)
    model = OpenMoeForCausalLM(config)
    logger.info(f"Finish init model with config:\n{config}", ranks=[0])

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Prepare tokenizer and dataloader
    dataset = RandomDataset(
        num_samples=args.batch_size * (args.warmup + args.active + 1) * dp_size,
        max_length=args.seq_length,
    )
    dataloader = plugin.prepare_dataloader(dataset, batch_size=args.batch_size)

    # Set optimizer
    optimizer = Adafactor(model.parameters(), weight_decay=0.01)

    model_numel = get_model_numel(model)
    performance_evaluator = PerformanceEvaluator(
        model_numel,
        enable_grad_checkpoint=True,
        ignore_steps=args.warmup,
        dp_world_size=dp_size,
    )

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _, dataloader, _ = booster.boost(model=model, optimizer=optimizer, dataloader=dataloader)
    use_pipeline = (isinstance(booster.plugin, MoeHybridParallelPlugin) and booster.plugin.pp_size > 1)
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    logger.info(f"Finish init booster", ranks=[0])

    # Start finetuning
    logger.info(f"Start finetuning", ranks=[0])
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
                    return_outputs=True,
                )
                # Backward and optimize
                if is_pp_last_stage:
                    loss = outputs["loss"]
                    pbar.set_postfix({"loss": loss.item()})
            else:
                # Forward pass
                data = move_to_cuda(data, torch.cuda.current_device())
                outputs = model(**data)
                loss = outputs["loss"]
                # Backward
                booster.backward(loss, optimizer)
                pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            optimizer.zero_grad()
            performance_evaluator.on_step_end(exmaple_data["input_ids"])
    performance_evaluator.on_fit_end()


if __name__ == "__main__":
    main()
