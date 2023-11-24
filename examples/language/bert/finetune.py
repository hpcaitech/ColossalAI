import argparse
from typing import Callable, List, Union

import evaluate
import torch
import torch.distributed as dist
import torch.nn as nn
from data import GLUEDataBuilder
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AlbertForSequenceClassification,
    AutoConfig,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

# ==============================
# Prepare Hyperparameters
# ==============================
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1

output_transform_fn = lambda x: x
criterion = lambda x: x.loss


def move_to_cuda(batch):
    return {k: v.cuda() for k, v in batch.items()}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    criterion,
    test_dataloader: Union[DataLoader, List[DataLoader]],
    num_labels: int,
    task_name: str,
    eval_splits: List[str],
    booster: Booster,
    coordinator: DistCoordinator,
):
    metric = evaluate.load("glue", task_name, process_id=coordinator.rank, num_process=coordinator.world_size)
    model.eval()

    def evaluate_subset(dataloader: DataLoader):
        use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
        is_pp_last_device = use_pipeline and booster.plugin.stage_manager.is_last_stage(ignore_chunk=True)

        accum_loss = torch.zeros(1, device=get_current_device())
        for batch in dataloader:
            batch = move_to_cuda(batch)
            labels = batch["labels"]
            if use_pipeline:
                pg_mesh = booster.plugin.pg_mesh
                pp_group = booster.plugin.pp_group
                current_pp_group_ranks = pg_mesh.get_ranks_in_group(pp_group)
                current_rank = dist.get_rank()
                batch = iter([batch])

                outputs = booster.execute_pipeline(batch, model, criterion, return_loss=True, return_outputs=True)

                if is_pp_last_device:
                    logits = outputs["outputs"]["logits"]
                    val_loss = outputs["loss"]
                    accum_loss.add_(val_loss)

                    if num_labels > 1:
                        preds = torch.argmax(logits, axis=1)
                    elif num_labels == 1:
                        preds = logits.squeeze()

                    dist.broadcast_object_list([preds, val_loss], src=current_pp_group_ranks[-1], group=pp_group)

                    metric.add_batch(predictions=preds, references=labels)
                elif current_rank in current_pp_group_ranks:
                    object_list = [None, None]
                    dist.broadcast_object_list(object_list, src=current_pp_group_ranks[-1], group=pp_group)

                    metric.add_batch(predictions=object_list[0].to(get_current_device()), references=labels)
                    accum_loss.add_(object_list[1].to(get_current_device()))

            else:
                batch = move_to_cuda(batch)
                outputs = model(**batch)
                val_loss, logits = outputs[:2]
                accum_loss.add_(val_loss)

                if num_labels > 1:
                    preds = torch.argmax(logits, axis=1)
                elif num_labels == 1:
                    preds = logits.squeeze()

                metric.add_batch(predictions=preds, references=labels)

        results = metric.compute()
        dist.all_reduce(accum_loss.div_(len(dataloader)))
        if coordinator.is_master() and results is not None:
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
    _criterion: Callable,
    lr_scheduler: LRScheduler,
    train_dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_device = use_pipeline and booster.plugin.stage_manager.is_last_stage(ignore_chunk=True)
    print_flag = (not use_pipeline and coordinator.is_master()) or (use_pipeline and is_pp_last_device)
    total_step = len(train_dataloader)

    model.train()
    optimizer.zero_grad()
    train_dataloader_iter = iter(train_dataloader)
    with tqdm(range(total_step), desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]", disable=not print_flag) as pbar:
        # Forward pass
        for _ in pbar:
            if use_pipeline:
                outputs = booster.execute_pipeline(
                    train_dataloader_iter, model, _criterion, optimizer, return_loss=True, return_outputs=True
                )
                # Backward and optimize
                if is_pp_last_device:
                    loss = outputs["loss"]
                    pbar.set_postfix({"loss": loss.item()})
            else:
                data = next(train_dataloader_iter)
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
        choices=["torch_ddp", "torch_ddp_fp16", "gemini", "low_level_zero", "hybrid_parallel"],
        help="plugin to use",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        help="bert or albert",
    )
    parser.add_argument("--target_f1", type=float, default=None, help="target f1 score. Raise exception if not reached")
    parser.add_argument("--use_lazy_init", type=bool, default=False, help="for initiating lazy init context")
    args = parser.parse_args()

    if args.model_type == "bert":
        model_name = "bert-base-uncased"
    elif args.model_type == "albert":
        model_name = "albert-xxlarge-v2"
    else:
        raise RuntimeError

    # ==============================
    # Launch Distributed Environment
    # ==============================
    colossalai.launch_from_torch(config={}, seed=42)
    coordinator = DistCoordinator()

    lr = LEARNING_RATE * coordinator.world_size

    # ==============================
    # Instantiate Plugin and Booster
    # ==============================
    booster_kwargs = {}
    if args.plugin == "torch_ddp_fp16":
        booster_kwargs["mixed_precision"] = "fp16"
    if args.plugin.startswith("torch_ddp"):
        plugin = TorchDDPPlugin()
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(initial_scale=2**5)
    elif args.plugin == "low_level_zero":
        plugin = LowLevelZeroPlugin(initial_scale=2**5)
    elif args.plugin == "hybrid_parallel":
        # modify the param accordingly for finetuning test cases
        plugin = HybridParallelPlugin(
            tp_size=1,
            pp_size=2,
            num_microbatches=None,
            pp_style="interleaved",
            num_model_chunks=2,
            microbatch_size=16,
            enable_all_optimization=True,
            zero_stage=1,
            precision="fp16",
            initial_scale=1,
        )

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

    cfg = AutoConfig.from_pretrained(model_name, num_labels=data_builder.num_labels)

    if model_name == "bert-base-uncased":
        model = BertForSequenceClassification.from_pretrained(model_name, config=cfg).cuda()
    elif model_name == "albert-xxlarge-v2":
        model = AlbertForSequenceClassification.from_pretrained(model_name, config=cfg)
    else:
        raise RuntimeError

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

    def _criterion(outputs, inputs):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss

    # ==============================
    # Boost with ColossalAI
    # ==============================
    model, optimizer, _criterion, _, lr_scheduler = booster.boost(
        model, optimizer, criterion=_criterion, lr_scheduler=lr_scheduler
    )

    # ==============================
    # Train model
    # ==============================
    for epoch in range(NUM_EPOCHS):
        train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, train_dataloader, booster, coordinator)

    results = evaluate_model(
        model,
        _criterion,
        test_dataloader,
        data_builder.num_labels,
        args.task,
        data_builder.eval_splits,
        booster,
        coordinator,
    )

    if coordinator.is_master():
        print(results)
        if args.target_f1 is not None and "f1" in results:
            assert results["f1"] >= args.target_f1, f'f1 score {results["f1"]} is lower than target {args.target_f1}'


if __name__ == "__main__":
    main()
