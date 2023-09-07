import argparse
from contextlib import nullcontext
from typing import Callable, List, Union

import evaluate
import torch
import torch.distributed as dist
import torch.nn as nn
from data import GLUEDataBuilder
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, LlamaForCausalLM, LlamaForSequenceClassification, get_linear_schedule_with_warmup

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

# ==============================
# Prepare Hyperparameters
# ==============================
NUM_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1

output_transform_fn = lambda x: x
criterion = lambda x: x.loss


def move_to_cuda(batch):
    return {k: v.cuda() for k, v in batch.items()}


def train_epoch(epoch: int, model: nn.Module, optimizer: Optimizer, _criterion: Callable, lr_scheduler: LRScheduler,
                train_dataloader: DataLoader, booster: Booster, coordinator: DistCoordinator):

    model.train()
    is_pp_last_stage = hasattr(
        booster.plugin,
        "stage_manager") and booster.plugin.stage_manager is not None and booster.plugin.stage_manager.is_last_stage()
    with tqdm(train_dataloader,
              desc=f'Epoch [{epoch + 1}/{NUM_EPOCHS}]',
              disable=not (coordinator.is_master() or is_pp_last_stage)) as pbar:
        for batch in pbar:
            # Forward pass
            batch = move_to_cuda(batch)
            if hasattr(booster.plugin, "stage_manager") and booster.plugin.stage_manager is not None:
                #TODO pass train_dataloader to execute_pipeline directly
                batch = iter([batch])
                outputs = booster.execute_pipeline(batch,
                                                   model,
                                                   _criterion,
                                                   optimizer,
                                                   return_loss=True,
                                                   return_outputs=True)
                # Backward and optimize
                if booster.plugin.stage_manager.is_last_stage():
                    loss = outputs['loss']
                    pbar.set_postfix({'loss': loss.item()})
            else:
                outputs = model(**batch)
                loss = _criterion(outputs, None)
                # Backward
                booster.backward(loss, optimizer)
                pbar.set_postfix({'loss': loss.item()})

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', default='super_natural_instructions', help="GLUE task to run")
    parser.add_argument('-p',
                        '--plugin',
                        type=str,
                        default='torch_ddp',
                        choices=['torch_ddp', 'torch_ddp_fp16', 'gemini', 'low_level_zero', 'hybrid_parallel'],
                        help="plugin to use")

    parser.add_argument('--model_path', type=str, help="model checkpoints path must be passed.")
    parser.add_argument('--target_f1', type=float, default=None, help="target f1 score. Raise exception if not reached")
    parser.add_argument('--use_lazy_init', type=bool, default=False, help="for initiating lazy init context")
    args = parser.parse_args()

    # ==============================
    # Launch Distributed Environment
    # ==============================
    colossalai.launch_from_torch(config={}, seed=42)
    coordinator = DistCoordinator()

    # local_batch_size = BATCH_SIZE // coordinator.world_size
    lr = LEARNING_RATE * coordinator.world_size

    # ==============================
    # Instantiate Plugin and Booster
    # ==============================
    booster_kwargs = {}
    if args.plugin == 'torch_ddp_fp16':
        booster_kwargs['mixed_precision'] = 'fp16'
    if args.plugin.startswith('torch_ddp'):
        plugin = TorchDDPPlugin()
    elif args.plugin == 'gemini':
        plugin = GeminiPlugin(initial_scale=2**5)
    elif args.plugin == 'low_level_zero':
        plugin = LowLevelZeroPlugin(initial_scale=2**5)
    elif args.plugin == 'hybrid_parallel':

        # modify the param accordingly for finetuning test cases
        plugin = HybridParallelPlugin(tp_size=4,
                                      pp_size=1,
                                      num_microbatches=None,
                                      microbatch_size=1,
                                      enable_jit_fused=False,
                                      zero_stage=0,
                                      precision='fp32',
                                      initial_scale=1)

    booster = Booster(plugin=plugin, **booster_kwargs)

    # ==============================
    # Prepare Dataloader
    # ==============================
    data_builder = GLUEDataBuilder(args.model_path,
                                   plugin,
                                   args.task,
                                   train_batch_size=BATCH_SIZE,
                                   eval_batch_size=BATCH_SIZE)
    train_dataloader = data_builder.train_dataloader()
    test_dataloader = data_builder.test_dataloader()

    # ====================================
    # Prepare model, optimizer
    # ====================================
    # bert pretrained model

    cfg = AutoConfig.from_pretrained(args.model_path)

    model = LlamaForCausalLM.from_pretrained(args.model_path, config=cfg).cuda()

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
    model, optimizer, _criterion, _, lr_scheduler = booster.boost(model,
                                                                  optimizer,
                                                                  criterion=_criterion,
                                                                  lr_scheduler=lr_scheduler)

    # ==============================
    # Train model
    # ==============================
    for epoch in range(NUM_EPOCHS):
        train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, train_dataloader, booster, coordinator)

    if coordinator.is_master():
        print(f"Finish finetuning")


if __name__ == '__main__':
    main()
