import datasets
import torch
import transformers
from huggingface_hub import snapshot_download
from model.modeling_openmoe import OpenMoeForCausalLM
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import T5Tokenizer, get_linear_schedule_with_warmup
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
    parser.add_argument("--model_name_or_path",
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
    parser.add_argument("--learning_rate",
                        type=float,
                        default=5e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--warmup_ratio",
                        type=float,
                        default=0.1,
                        help="Ratio of warmup steps against total training steps.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Launch ColossalAI
    colossalai.launch_from_torch(config={}, seed=args.seed)
    coordinator = DistCoordinator()
    world_size = coordinator.world_size

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
    repo_name = "hpcaitech/openmoe-" + args.model_name_or_path
    config = LlamaConfig.from_pretrained(repo_name)
    with skip_init():
        model = OpenMoeForCausalLM(config)
    ckpt_path = snapshot_download(repo_name)
    MoeCheckpintIO().load_model(model, ckpt_path + "/pytorch_model.bin")
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
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=(args.learning_rate * world_size),
                                 weight_decay=args.weight_decay)

    # Set lr scheduler
    total_steps = len(dataloader) * args.num_epoch
    num_warmup_steps = int(args.warmup_ratio * total_steps)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=len(dataloader) * args.num_epoch)

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(model=model,
                                                                  optimizer=optimizer,
                                                                  dataloader=dataloader,
                                                                  lr_scheduler=lr_scheduler)
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
                lr_scheduler.step()

                # Print batch loss
                pbar.set_postfix({'loss': loss.item()})

    # Finish training and evaluate
    logger.info(f"Finish finetuning", ranks=[0])
    booster.save_model(model, args.output_path)
    logger.info(f"Saving model checkpoint to {args.output_path}", ranks=[0])


if __name__ == "__main__":
    main()
