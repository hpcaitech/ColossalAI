import os

import datasets
import torch
import transformers
from huggingface_hub import snapshot_download
from model.modeling_openmoe import OpenMoeForCausalLM
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Adafactor, T5Tokenizer
from transformers.models.llama import LlamaConfig

import colossalai
from colossalai import get_default_parser
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.context import MOE_CONTEXT
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.layer.moe import MoeCheckpintIO
from colossalai.nn.layer.moe.utils import skip_init
from colossalai.utils import get_current_device


def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def load_ckpt(repo_name: str, model: OpenMoeForCausalLM):
    ckpt_path = snapshot_download(repo_name)
    # single ckpt
    if os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin")):
        ckpt_path = os.path.join(ckpt_path, "pytorch_model.bin")
    # shard ckpt
    elif os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin.index.json")):
        ckpt_path = os.path.join(ckpt_path, "pytorch_model.bin.index.json")
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")
    MoeCheckpintIO().load_model(model, ckpt_path)


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
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.input_ids[idx]
        }


def parse_args():
    parser = get_default_parser()
    parser.add_argument("--model_name",
                        type=str,
                        default="base",
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--output_path",
                        type=str,
                        default="./output_model.bin",
                        help="The path of your saved model after finetuning.")
    parser.add_argument("--num_epoch", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=4,
                        help="Batch size (per dp group) for the training dataloader.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    # loss
    parser.add_argument("--router_aux_loss_factor", type=float, default=0.01, help="router_aux_loss_factor.")
    parser.add_argument("--router_z_loss_factor", type=float, default=0.0001, help="router_z_loss_factor.")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="label_smoothing.")
    parser.add_argument("--z_loss_factor", type=float, default=0.0001, help="z_loss_factor.")
    # optim
    parser.add_argument("--decay_rate", type=float, default=-0.8, help="adafactor optim decay rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Launch ColossalAI
    colossalai.launch_from_torch(config={}, seed=args.seed)
    coordinator = DistCoordinator()

    # Set up moe
    MOE_CONTEXT.setup(seed=42, parallel="EP")

    # Manage loggers
    disable_existing_loggers()
    logger = get_dist_logger()
    if coordinator.is_master():
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Build OpenMoe model
    repo_name = "hpcaitech/openmoe-" + args.model_name
    config = LlamaConfig.from_pretrained(repo_name)
    setattr(config, "router_aux_loss_factor", args.router_aux_loss_factor)
    setattr(config, "router_z_loss_factor", args.router_z_loss_factor)
    setattr(config, "label_smoothing", args.label_smoothing)
    setattr(config, "z_loss_factor", args.z_loss_factor)
    with skip_init():
        model = OpenMoeForCausalLM(config)
    load_ckpt(repo_name, model)
    logger.info(f"Finish init model with config:\n{config}", ranks=[0])

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Set plugin
    booster_kwargs = {}
    plugin = LowLevelZeroPlugin(initial_scale=2**5, stage=2)
    logger.info(f"Set plugin as {plugin}", ranks=[0])

    # Prepare tokenizer and dataloader
    tokenizer = T5Tokenizer.from_pretrained("google/umt5-small")
    dataset = RandomDataset()
    dataloader = plugin.prepare_dataloader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Set optimizer
    optimizer = Adafactor(model.parameters(), decay_rate=args.decay_rate, weight_decay=args.weight_decay)

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _, dataloader, _ = booster.boost(model=model, optimizer=optimizer, dataloader=dataloader)
    logger.info(f"Finish init booster", ranks=[0])

    # Start finetuning
    logger.info(f"Start finetuning", ranks=[0])
    for epoch in range(args.num_epoch):
        model.train()
        with tqdm(dataloader, desc=f'Epoch [{epoch + 1}]', disable=not coordinator.is_master()) as pbar:
            for batch in pbar:
                # Forward
                optimizer.zero_grad()
                batch = move_to_cuda(batch, torch.cuda.current_device())

                outputs = model(use_cache=False, chunk_head=True, **batch)
                loss = outputs['loss']

                # Backward
                booster.backward(loss, optimizer)
                optimizer.step()

                # Print batch loss
                pbar.set_postfix({'loss': loss.item()})

    # Finish training and evaluate
    logger.info(f"Finish finetuning", ranks=[0])
    booster.save_model(model, args.output_path)
    logger.info(f"Saving model checkpoint to {args.output_path}", ranks=[0])


if __name__ == "__main__":
    main()
