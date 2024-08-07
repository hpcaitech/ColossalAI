from contextlib import nullcontext

import datasets
import torch
import transformers
from args import parse_demo_args
from data import NetflixDataset, netflix_collator
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, OPTForCausalLM, get_linear_schedule_with_warmup
from transformers.utils.versions import require_version

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam

require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")
require_version("transformers>=4.20.0", "To fix: pip install -r requirements.txt")

output_transform_fn = lambda x: x
criterion = lambda x: x.loss


def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, dataloader, booster, coordinator):
    torch.cuda.synchronize()

    use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    total_step = len(dataloader)

    model.train()
    optimizer.zero_grad()
    dataloader = iter(dataloader)
    with tqdm(
        range(total_step), desc=f"Epoch [{epoch + 1}]", disable=not (coordinator.is_master() or is_pp_last_stage)
    ) as pbar:
        # Forward pass
        for _ in pbar:
            if use_pipeline:
                outputs = booster.execute_pipeline(dataloader, model, _criterion, optimizer, return_loss=True)
                # Backward and optimize
                if is_pp_last_stage:
                    loss = outputs["loss"]
                    pbar.set_postfix({"loss": loss.item()})
            else:
                data = next(dataloader)
                data = move_to_cuda(data)
                outputs = model(**data)
                loss = _criterion(outputs, None)
                # Backward
                booster.backward(loss, optimizer)
                pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()


def main():
    args = parse_demo_args()

    # Launch ColossalAI
    colossalai.launch_from_torch(seed=args.seed)
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

    # Set plugin
    booster_kwargs = {}
    if args.plugin == "torch_ddp_fp16":
        booster_kwargs["mixed_precision"] = "fp16"
    if args.plugin.startswith("torch_ddp"):
        plugin = TorchDDPPlugin()
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(offload_optim_frac=1.0, pin_memory=True, initial_scale=2**5)
    elif args.plugin == "low_level_zero":
        plugin = LowLevelZeroPlugin(initial_scale=2**5)
    elif args.plugin == "hybrid_parallel":
        # modify the param accordingly for finetuning test cases
        plugin = HybridParallelPlugin(
            tp_size=2,
            pp_size=2,
            num_microbatches=2,
            enable_all_optimization=True,
            zero_stage=0,
            precision="fp16",
            initial_scale=1,
        )

    logger.info(f"Set plugin as {args.plugin}", ranks=[0])

    # Build OPT model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    # Build OPT model
    init_ctx = (
        LazyInitContext(default_device=get_accelerator().get_current_device())
        if isinstance(plugin, (GeminiPlugin, HybridParallelPlugin))
        else nullcontext()
    )
    with init_ctx:
        model = OPTForCausalLM.from_pretrained(args.model_name_or_path, config=config)
    logger.info(f"Finish loading model from {args.model_name_or_path}", ranks=[0])

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Prepare tokenizer and dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = NetflixDataset(tokenizer)
    dataloader = plugin.prepare_dataloader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=netflix_collator
    )

    # Set optimizer
    optimizer = HybridAdam(model.parameters(), lr=(args.learning_rate * world_size), weight_decay=args.weight_decay)

    # Set lr scheduler
    total_steps = len(dataloader) * args.num_epoch
    num_warmup_steps = int(args.warmup_ratio * total_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=len(dataloader) * args.num_epoch
    )

    # Define criterion
    def _criterion(outputs, inputs):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _criterion, dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, dataloader=dataloader, criterion=_criterion, lr_scheduler=lr_scheduler
    )

    # Start finetuning
    logger.info(f"Start finetuning", ranks=[0])
    for epoch in range(args.num_epoch):
        train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, dataloader, booster, coordinator)

    # Finish training and evaluate
    logger.info(f"Finish finetuning", ranks=[0])
    booster.save_model(model, args.output_path, shard=True)
    logger.info(f"Saving model checkpoint to {args.output_path}", ranks=[0])


if __name__ == "__main__":
    main()
