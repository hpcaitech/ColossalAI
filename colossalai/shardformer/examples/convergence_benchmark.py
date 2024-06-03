import argparse
import math
from typing import Any, List, Union

import evaluate
import torch
import torch.distributed as dist
from data import GLUEDataBuilder
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, get_linear_schedule_with_warmup

import colossalai
from colossalai.cluster import DistCoordinator
from colossalai.shardformer import ShardConfig, ShardFormer


def to_device(x: Any, device: torch.device) -> Any:
    def _to(t: Any):
        if isinstance(t, torch.Tensor):
            return t.to(device)
        return t

    return tree_map(_to, x)


def train(args):
    colossalai.launch_from_torch(seed=42)
    coordinator = DistCoordinator()

    # prepare for data and dataset
    data_builder = GLUEDataBuilder(
        model_name_or_path=args.pretrain,
        task_name=args.task,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
    )
    train_dataloader = data_builder.train_dataloader()
    test_dataloader = data_builder.test_dataloader()

    if args.model == "bert":
        cfg = BertConfig.from_pretrained(args.pretrain, num_labels=data_builder.num_labels)
        model = BertForSequenceClassification.from_pretrained(args.pretrain, config=cfg)

    model.to(torch.cuda.current_device())

    # if multiple GPUs, shard the model
    if dist.get_world_size() > 1:
        tp_group = dist.new_group(backend="nccl")
        shard_config = ShardConfig(
            tensor_parallel_process_group=tp_group, enable_tensor_parallelism=True, enable_all_optimization=True
        )
        shard_former = ShardFormer(shard_config=shard_config)
        model, _ = shard_former.optimize(model)

    optim = Adam(model.parameters(), lr=args.lr)
    num_update_steps_per_epoch = len(train_dataloader) // args.accumulation_steps
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
    lr_scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=math.ceil(max_steps * args.warmup_fraction),
        num_training_steps=max_steps,
    )
    fit(
        model,
        optim,
        lr_scheduler,
        train_dataloader,
        args.max_epochs,
        args.accumulation_steps,
        args.batch_size,
        coordinator,
    )
    results = evaluate_model(
        model, test_dataloader, data_builder.num_labels, args.task, data_builder.eval_splits, coordinator
    )
    if coordinator.is_master():
        print(results)
        if args.target_f1 is not None and "f1" in results:
            assert results["f1"] >= args.target_f1, f'f1 score {results["f1"]} is lower than target {args.target_f1}'


def fit(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler,
    train_dataloader,
    max_epochs,
    accumulation_steps,
    batch_size,
    coordinator,
):
    step_bar = tqdm(
        range(len(train_dataloader) // accumulation_steps * max_epochs),
        desc=f"steps",
        disable=not coordinator.is_master(),
    )
    total_loss = 0
    for epoch in range(max_epochs):
        model.train()
        for batch_id, batch in enumerate(train_dataloader):
            batch = to_device(batch, torch.cuda.current_device())
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / accumulation_steps
            loss.backward()
            total_loss += loss.item()
            if (batch_id + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_bar.set_postfix(
                    {"epoch": epoch, "loss": total_loss / batch_size, "lr": scheduler.get_last_lr()[0]}
                )
                total_loss = 0
                step_bar.update()


# evaluate
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_dataloader: Union[DataLoader, List[DataLoader]],
    num_labels: int,
    task_name: str,
    eval_splits: List[str],
    coordinator: DistCoordinator,
):
    metric = evaluate.load("glue", task_name, process_id=coordinator.rank, num_process=coordinator.world_size)
    model.eval()

    def evaluate_subset(dataloader: DataLoader):
        accum_loss = torch.zeros(1, device=torch.cuda.current_device())
        for batch in dataloader:
            batch = to_device(batch, torch.cuda.current_device())
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
        if coordinator.is_master():
            results["loss"] = accum_loss.item() / (len(dataloader) * dataloader.batch_size)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", default="mrpc", help="GLUE task to run")
    parser.add_argument("--model", type=str, default="bert")
    parser.add_argument("--pretrain", type=str, default="bert-base-uncased")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.4e-5)
    parser.add_argument("--fused_layernorm", type=bool, default=False)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_fraction", type=float, default=0.03)
    parser.add_argument("--target_f1", type=float, default=None)
    args = parser.parse_args()
    train(args)
