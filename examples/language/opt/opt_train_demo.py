import time

import torch
import datasets
import transformers
from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers.utils.versions import require_version
from tqdm import tqdm

import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.tensor import ProcessGroup, ShardSpec
from colossalai.utils import get_current_device
from colossalai.zero import ColoInitContext
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator

from args import parse_demo_args
from data import NetflixDataset, netflix_collator

require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")
require_version("transformers>=4.20.0", "To fix: pip install -r requirements.txt")


def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator):
        
    torch.cuda.synchronize()
    model.train()

    with tqdm(dataloader, desc=f'Epoch [{epoch + 1}]', disable=not coordinator.is_master()) as pbar:
        
        for batch in pbar:

            # Forward
            optimizer.zero_grad()
            batch = move_to_cuda(batch, torch.cuda.current_device())
            
            outputs = model(use_cache=False, **batch)
            loss = outputs['loss']

            # Backward
            booster.backward(loss, optimizer)
            optimizer.step()
            lr_scheduler.step()

            # Print batch loss
            pbar.set_postfix({'loss': loss.item()})


def main():

    args = parse_demo_args()

    # Launch ColossalAI
    colossalai.launch_from_torch(config={}, seed=args.seed)
    coordinator = DistCoordinator()
    world_size = coordinator.world_size

    # Manage loggers
    disable_existing_loggers()
    logger = get_dist_logger()
    if coordinator.is_master():
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    # Build OPT model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = OPTForCausalLM.from_pretrained(args.model_name_or_path, config=config)
    logger.info(f"Finish loading model from {args.model_name_or_path}", ranks=[0])

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Set plugin
    booster_kwargs = {}
    if args.plugin == 'torch_ddp_fp16':
        booster_kwargs['mixed_precision'] = 'fp16'
    if args.plugin.startswith('torch_ddp'):
        plugin = TorchDDPPlugin()
    elif args.plugin == 'gemini':
        plugin = GeminiPlugin(device=get_current_device(),
                        placement_policy='cpu',
                        pin_memory=True,
                        strict_ddp_mode=True,
                        initial_scale=2**5)
    elif args.plugin == 'low_level_zero':
        plugin = LowLevelZeroPlugin(initial_scale=2**5)
    logger.info(f"Set plugin as {args.plugin}", ranks=[0])

    # Prepare tokenizer and dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)   
    dataset = NetflixDataset(tokenizer)
    dataloader = plugin.prepare_dataloader(dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=netflix_collator)
    
    # Set optimizer
    optimizer = HybridAdam(model.parameters(), 
                           lr=(args.learning_rate * world_size),
                           weight_decay=args.weight_decay)

    # Set lr scheduler
    total_steps = len(dataloader) * args.num_epoch
    num_warmup_steps = int(args.warmup_ratio * total_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(dataloader) * args.num_epoch
    )

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(model=model, 
                                                                  optimizer=optimizer, 
                                                                  dataloader=dataloader, 
                                                                  lr_scheduler=lr_scheduler)

    # Start finetuning
    logger.info(f"Start finetuning", ranks=[0])
    for epoch in range(args.num_epoch):
        train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator)

    # Finish training and evaluate
    logger.info(f"Finish finetuning", ranks=[0])
    booster.save_model(model, args.output_path)
    logger.info(f"Saving model checkpoint to {args.output_path}", ranks=[0])


if __name__ == "__main__":
    main()
