from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from args import parse_demo_args
from data import BeansDataset, beans_collator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam


def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def run_forward_backward(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: Callable[[Any, Any], torch.Tensor],
    data_iter: Iterator,
    booster: Booster,
):
    if optimizer is not None:
        optimizer.zero_grad()
    if isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1:
        # run pipeline forward backward when enabling pp in hybrid parallel plugin
        output_dict = booster.execute_pipeline(
            data_iter, model, criterion, optimizer, return_loss=True, return_outputs=True
        )
        loss, outputs = output_dict["loss"], output_dict["outputs"]
    else:
        batch = next(data_iter)
        batch = move_to_cuda(batch, torch.cuda.current_device())
        outputs = model(**batch)
        loss = criterion(outputs, None)
        if optimizer is not None:
            booster.backward(loss, optimizer)

    return loss, outputs


def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    criterion: Callable[[Any, Any], torch.Tensor],
    lr_scheduler: LRScheduler,
    dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    torch.cuda.synchronize()

    num_steps = len(dataloader)
    data_iter = iter(dataloader)
    enable_pbar = coordinator.is_master()
    if isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1:
        # when using pp, only the last stage of master pipeline (dp_rank and tp_rank are both zero) shows pbar
        tp_rank = dist.get_rank(booster.plugin.tp_group)
        dp_rank = dist.get_rank(booster.plugin.dp_group)
        enable_pbar = tp_rank == 0 and dp_rank == 0 and booster.plugin.stage_manager.is_last_stage()

    model.train()

    with tqdm(range(num_steps), desc=f"Epoch [{epoch + 1}]", disable=not enable_pbar) as pbar:
        for _ in pbar:
            loss, _ = run_forward_backward(model, optimizer, criterion, data_iter, booster)
            optimizer.step()
            lr_scheduler.step()

            # Print batch loss
            if enable_pbar:
                pbar.set_postfix({"loss": loss.item()})


@torch.no_grad()
def evaluate_model(
    epoch: int,
    model: nn.Module,
    criterion: Callable[[Any, Any], torch.Tensor],
    eval_dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    torch.cuda.synchronize()
    model.eval()
    accum_loss = torch.zeros(1, device=torch.cuda.current_device())
    total_num = torch.zeros(1, device=torch.cuda.current_device())
    accum_correct = torch.zeros(1, device=torch.cuda.current_device())

    for batch in eval_dataloader:
        batch = move_to_cuda(batch, torch.cuda.current_device())
        loss, outputs = run_forward_backward(model, None, criterion, iter([batch]), booster)

        to_accum = True
        if isinstance(booster.plugin, HybridParallelPlugin):
            # when using hybrid parallel, loss is only collected from last stage of pipeline with tp_rank == 0
            to_accum = to_accum and (dist.get_rank(booster.plugin.tp_group) == 0)
            if booster.plugin.pp_size > 1:
                to_accum = to_accum and booster.plugin.stage_manager.is_last_stage()

        if to_accum:
            accum_loss += loss / len(eval_dataloader)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=1)

            labels = batch["labels"]
            total_num += batch["labels"].shape[0]
            accum_correct += torch.sum(preds == labels)

    dist.all_reduce(accum_loss)
    dist.all_reduce(total_num)
    dist.all_reduce(accum_correct)
    avg_loss = "{:.4f}".format(accum_loss.item())
    accuracy = "{:.4f}".format(accum_correct.item() / total_num.item())
    if coordinator.is_master():
        print(
            f"Evaluation result for epoch {epoch + 1}: \
                average_loss={avg_loss}, \
                accuracy={accuracy}."
        )


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
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # Reset tp_size and pp_size to 1 if not using hybrid parallel.
    if args.plugin != "hybrid_parallel":
        args.tp_size = 1
        args.pp_size = 1

    # Prepare Dataset
    image_processor = ViTImageProcessor.from_pretrained(args.model_name_or_path)
    train_dataset = BeansDataset(image_processor, args.tp_size, split="train")
    eval_dataset = BeansDataset(image_processor, args.tp_size, split="validation")
    num_labels = train_dataset.num_labels

    # Load pretrained ViT model
    config = ViTConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = num_labels
    config.id2label = {str(i): c for i, c in enumerate(train_dataset.label_names)}
    config.label2id = {c: str(i) for i, c in enumerate(train_dataset.label_names)}
    model = ViTForImageClassification.from_pretrained(
        args.model_name_or_path, config=config, ignore_mismatched_sizes=True
    )
    logger.info(f"Finish loading model from {args.model_name_or_path}", ranks=[0])

    # Enable gradient checkpointing
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

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
        plugin = HybridParallelPlugin(
            tp_size=args.tp_size,
            pp_size=args.pp_size,
            num_microbatches=None,
            microbatch_size=1,
            enable_all_optimization=True,
            precision="fp16",
            initial_scale=1,
        )
    else:
        raise ValueError(f"Plugin with name {args.plugin} is not supported!")
    logger.info(f"Set plugin as {args.plugin}", ranks=[0])

    # Prepare dataloader
    train_dataloader = plugin.prepare_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=beans_collator
    )
    eval_dataloader = plugin.prepare_dataloader(
        eval_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=beans_collator
    )

    # Set optimizer
    optimizer = HybridAdam(model.parameters(), lr=(args.learning_rate * world_size), weight_decay=args.weight_decay)

    # Set criterion (loss function)
    def criterion(outputs, inputs):
        return outputs.loss

    # Set lr scheduler
    total_steps = len(train_dataloader) * args.num_epoch
    num_warmup_steps = int(args.warmup_ratio * total_steps)
    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer, total_steps=(len(train_dataloader) * args.num_epoch), warmup_steps=num_warmup_steps
    )

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _criterion, train_dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, criterion=criterion, dataloader=train_dataloader, lr_scheduler=lr_scheduler
    )

    # Finetuning
    logger.info(f"Start finetuning", ranks=[0])
    for epoch in range(args.num_epoch):
        train_epoch(epoch, model, optimizer, criterion, lr_scheduler, train_dataloader, booster, coordinator)
        evaluate_model(epoch, model, criterion, eval_dataloader, booster, coordinator)
    logger.info(f"Finish finetuning", ranks=[0])

    # Save the finetuned model
    booster.save_model(model, args.output_path, shard=True)
    logger.info(f"Saving model checkpoint to {args.output_path}", ranks=[0])


if __name__ == "__main__":
    main()
