import argparse
from typing import List, Union

import datasets
import torch
import torch.distributed as dist
import torch.nn as nn
from data import GLUEDataBuilder
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, BertForSequenceClassification, get_linear_schedule_with_warmup

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam

# ==============================
# Prepare Hyperparameters
# ==============================
NUM_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1


def move_to_cuda(batch):
    return {k: v.cuda() for k, v in batch.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_dataloader: Union[DataLoader, List[DataLoader]],
    num_labels: int,
    task_name: str,
    eval_splits: List[str],
    coordinator: DistCoordinator,
):
    metric = datasets.load_metric("glue", task_name, process_id=coordinator.rank, num_process=coordinator.world_size)
    model.eval()

    def evaluate_subset(dataloader: DataLoader):
        accum_loss = torch.zeros(1, device=get_accelerator().get_current_device())
        for batch in dataloader:
            batch = move_to_cuda(batch)
            outputs = model(**batch)
            val_loss, logits = outputs[:2]
            accum_loss.add_(val_loss)

            if num_labels > 1:
                preds = torch.argmax(logits, axis=1)
            elif num_labels == 1:
                preds = logits.squeeze()

            labels = batch["labels"]

            metric.add_batch(predictions=preds, references=labels)

        results = metric.compute()
        dist.all_reduce(accum_loss.div_(len(dataloader)))
        if coordinator.is_master():
            results["loss"] = accum_loss.item() / coordinator.world_size
        return results

    if isinstance(test_dataloader, DataLoader):
        return evaluate_subset(test_dataloader)
    else:
        assert len(test_dataloader) == len(eval_splits)
        final_results = {}
        for split, sub_loader in zip(eval_splits, test_dataloader):
            results = evaluate_subset(sub_loader)
            final_results.update({f"{k}_{split}": v for k, v in results.items()})
        return final_results


def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler,
    train_dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    model.train()
    with tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]", disable=not coordinator.is_master()) as pbar:
        for batch in pbar:
            # Forward pass
            batch = move_to_cuda(batch)
            outputs = model(**batch)
            loss = outputs[0]

            # Backward and optimize
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            # Print log info
            pbar.set_postfix({"loss": loss.item()})


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", default="mrpc", help="GLUE task to run")
    parser.add_argument(
        "-p",
        "--plugin",
        type=str,
        default="torch_ddp",
        choices=["torch_ddp", "torch_ddp_fp16", "gemini", "low_level_zero"],
        help="plugin to use",
    )
    parser.add_argument("--target_f1", type=float, default=None, help="target f1 score. Raise exception if not reached")
    args = parser.parse_args()

    # ==============================
    # Launch Distributed Environment
    # ==============================
    colossalai.launch_from_torch(seed=42)
    coordinator = DistCoordinator()

    # local_batch_size = BATCH_SIZE // coordinator.world_size
    lr = LEARNING_RATE * coordinator.world_size
    model_name = "bert-base-uncased"

    # ==============================
    # Instantiate Plugin and Booster
    # ==============================
    booster_kwargs = {}
    if args.plugin == "torch_ddp_fp16":
        booster_kwargs["mixed_precision"] = "fp16"
    if args.plugin.startswith("torch_ddp"):
        plugin = TorchDDPPlugin()
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(placement_policy="static", strict_ddp_mode=True, initial_scale=2**5)
    elif args.plugin == "low_level_zero":
        plugin = LowLevelZeroPlugin(initial_scale=2**5)

    booster = Booster(plugin=plugin, **booster_kwargs)

    # ==============================
    # Prepare Dataloader
    # ==============================
    data_builder = GLUEDataBuilder(
        model_name, plugin, args.task, train_batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE
    )
    train_dataloader = data_builder.train_dataloader()
    test_dataloader = data_builder.test_dataloader()

    # ====================================
    # Prepare model, optimizer
    # ====================================
    # bert pretrained model
    config = AutoConfig.from_pretrained(model_name, num_labels=data_builder.num_labels)
    model = BertForSequenceClassification.from_pretrained(model_name, config=config)

    # optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = HybridAdam(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    # lr scheduler
    total_steps = len(train_dataloader) * NUM_EPOCHS
    num_warmup_steps = int(WARMUP_FRACTION * total_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    # ==============================
    # Boost with ColossalAI
    # ==============================
    model, optimizer, _, _, lr_scheduler = booster.boost(model, optimizer, lr_scheduler=lr_scheduler)

    # ==============================
    # Train model
    # ==============================
    for epoch in range(NUM_EPOCHS):
        train_epoch(epoch, model, optimizer, lr_scheduler, train_dataloader, booster, coordinator)

    results = evaluate(
        model, test_dataloader, data_builder.num_labels, args.task, data_builder.eval_splits, coordinator
    )

    if coordinator.is_master():
        print(results)
        if args.target_f1 is not None and "f1" in results:
            assert results["f1"] >= args.target_f1, f'f1 score {results["f1"]} is lower than target {args.target_f1}'


if __name__ == "__main__":
    main()
