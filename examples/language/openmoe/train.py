import argparse
import os

import datasets
import torch
import torch.distributed as dist
import transformers
from huggingface_hub import snapshot_download
from model.modeling_openmoe import OpenMoeForCausalLM, set_openmoe_args
from model.openmoe_policy import OpenMoeForCausalLMPolicy
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Adafactor, T5Tokenizer
from transformers.models.llama import LlamaConfig

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.moe.layers import apply_load_balance
from colossalai.moe.manager import MOE_MANAGER
from colossalai.moe.utils import skip_init
from colossalai.utils import get_current_device


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
    def __init__(self, num_samples: int = 1000, max_length: int = 2048, vocab_size: int = 32000, tokenizer=None):
        """
        A random dataset

        You can use tokenizer to process your own data
        Example:
            self.input_ids = []
            self.attention_mask = []
            data = your_data()
            data = shuffle(data)
            for text in data:
                # text is a str
                encode = tokenizer(
                    "<pad>" + text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length")
                self.input_ids.append(encode["input_ids"])
                self.attention_mask.append(encode["attention_mask"])
            self.input_ids = torch.cat(self.input_ids, dim=0).to(get_current_device())
            self.attention_mask = torch.cat(self.attention_mask, dim=0).to(get_current_device())
        """
        # TODO: use distributed sampler
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
        default="base",
        choices=["base", "8b", "test"],
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output_model.bin",
        help="The path of your saved model after finetuning.",
    )
    parser.add_argument("--num_epoch", type=int, default=10, help="Number of epochs.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (per dp group) for the training dataloader.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--plugin",
        type=str,
        default="hybrid",
        help="parallel plugin",
        choices=["ep", "ep_zero", "hybrid"],
    )

    # optim
    parser.add_argument("--decay_rate", type=float, default=-0.8, help="adafactor optim decay rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")

    # zero stage for all plugins
    parser.add_argument("--zero_stage", type=int, default=2, help="zero stage in hybrid plugin")

    # ep zero plugin
    parser.add_argument("--extra_dp_size", type=int, default=1, help="ep zero's moe dp size")

    # hybrid plugin
    parser.add_argument("--pp_size", type=int, default=2, help="pp size")
    parser.add_argument("--dp_size", type=int, default=1, help="dp size")
    parser.add_argument("--ep_size", type=int, default=2, help="ep size")
    parser.add_argument("--microbatch_size", type=int, default=1, help="microbatch size")

    # kernel
    parser.add_argument(
        "--use_kernel",
        action="store_true",
        help="Use kernel optim. Need to install flash attention and triton to enable all kernel optimizations.",
    )
    parser.add_argument(
        "--use_layernorm_kernel",
        action="store_true",
        help="Use layernorm kernel. Need to install apex.",
    )

    # loss
    parser.add_argument(
        "--router_aux_loss_factor",
        type=float,
        default=0.01,
        help="router_aux_loss_factor.",
    )
    parser.add_argument(
        "--router_z_loss_factor",
        type=float,
        default=0.0001,
        help="router_z_loss_factor.",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="label_smoothing.")
    parser.add_argument("--z_loss_factor", type=float, default=0.0001, help="z_loss_factor.")

    # load balance
    parser.add_argument("--load_balance", action="store_true", help="moe load balance")
    parser.add_argument("--load_balance_interval", type=int, default=1000, help="moe load balance interval")

    # overlap
    parser.add_argument("--comm_overlap", action="store_true", help="moe comm overlap")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Launch ColossalAI
    colossalai.launch_from_torch(config={}, seed=args.seed)
    coordinator = DistCoordinator()
    test_mode = args.model_name == "test"

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
    hybrid_dict = {
        "tp_size": 1,
        "custom_policy": OpenMoeForCausalLMPolicy(),
        "enable_fused_normalization": args.use_layernorm_kernel,
        "enable_jit_fused": args.use_kernel,
        "precision": "bf16",
        "zero_stage": args.zero_stage,
    }
    mgr_dict = {
        "seed": 42,
    }
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
            extra_dp_size=args.extra_dp_size,
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
    logger.info(f"Set plugin as {plugin}", ranks=[0])

    # Build OpenMoe model
    if test_mode:
        config = LlamaConfig.from_pretrained("hpcaitech/openmoe-base")
        config.hidden_size = 64
        config.intermediate_size = 128
        config.vocab_size = 32000
    else:
        repo_name = "hpcaitech/openmoe-" + args.model_name
        config = LlamaConfig.from_pretrained(repo_name)
    set_openmoe_args(
        config,
        num_experts=config.num_experts,
        moe_layer_interval=config.moe_layer_interval,
        router_aux_loss_factor=args.router_aux_loss_factor,
        router_z_loss_factor=args.router_z_loss_factor,
        z_loss_factor=args.z_loss_factor,
        enable_load_balance=args.load_balance,
        enable_comm_overlap=args.comm_overlap,
        enable_kernel=args.use_kernel,
    )
    with skip_init():
        model = OpenMoeForCausalLM(config)
    logger.info(f"Finish init model with config:\n{config}", ranks=[0])

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Prepare tokenizer and dataloader
    tokenizer = T5Tokenizer.from_pretrained("google/umt5-small")
    dataset = RandomDataset(num_samples=1000 if not test_mode else 20, tokenizer=tokenizer)
    dataloader = plugin.prepare_dataloader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Set optimizer
    optimizer = Adafactor(model.parameters(), decay_rate=args.decay_rate, weight_decay=args.weight_decay)

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _, dataloader, _ = booster.boost(model=model, optimizer=optimizer, dataloader=dataloader)
    if not test_mode:
        load_ckpt(repo_name, model, booster)
    use_pipeline = isinstance(booster.plugin, MoeHybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    logger.info(f"Finish init booster", ranks=[0])

    # Start finetuning
    logger.info(f"Start finetuning", ranks=[0])
    for epoch in range(args.num_epoch):
        model.train()
        train_dataloader_iter = iter(dataloader)
        total_len = len(train_dataloader_iter)
        with tqdm(
            range(total_len),
            desc=f"Epoch [{epoch + 1}/{args.num_epoch}]",
            disable=not coordinator.is_master(),
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
                        return_outputs=True,
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

                # Apply load balance
                if args.load_balance and args.load_balance_interval > 0 and step % args.load_balance_interval == 0:
                    coordinator.print_on_master(f"Apply load balance")
                    apply_load_balance(model, optimizer)

    # Finish training and evaluate
    logger.info(f"Finish finetuning", ranks=[0])
    booster.save_model(model, args.output_path)
    logger.info(f"Saving model checkpoint to {args.output_path}", ranks=[0])


if __name__ == "__main__":
    main()
