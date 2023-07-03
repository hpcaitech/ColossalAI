# TODO: checkpoint
# TODO: tensorboard
# TODO: wandb

import argparse
import resource
from contextlib import nullcontext
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision
from tqdm import tqdm
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

MODEL_CONFIGS = {
    '7b': LlamaConfig(),
    '13b': LlamaConfig(hidden_size=5120, intermediate_size=13760, num_hidden_layers=40, num_attention_heads=40),
    '30b': LlamaConfig(hidden_size=6656, intermediate_size=17888, num_hidden_layers=60, num_attention_heads=52),
    '65b': LlamaConfig(hidden_size=8192, intermediate_size=22016, num_hidden_layers=80, num_attention_heads=64),
}


def get_model_numel(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f'{numel / B:.2f} B'
    elif numel >= M:
        return f'{numel / M:.2f} M'
    elif numel >= K:
        return f'{numel / K:.2f} K'
    else:
        return f'{numel}'


def tokenize_batch(batch, tokenizer: Optional[LlamaTokenizer] = None, max_length: int = 2048):
    texts = [sample['text'] for sample in batch]
    data = tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
    data['labels'] = data['input_ids'].clone()
    return data


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='7b', help='Model configuration')
    parser.add_argument('-p',
                        '--plugin',
                        choices=['gemini', 'gemini_cpu', 'zero2', 'zero2_cpu'],
                        default='gemini',
                        help='Choose which plugin to use')
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        default='togethercomputer/RedPajama-Data-1T-Sample',
                        help='Data set path')
    parser.add_argument('-e', '--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Local batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('-w', '--weigth_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('-s', '--warmup_steps', type=int, default=2000, help='Warmup steps')
    parser.add_argument('-g', '--grad_checkpoint', action='store_true', help='Use gradient checkpointing')
    parser.add_argument('-l', '--max_length', type=int, default=2048, help='Max sequence length')
    parser.add_argument('-x', '--mixed_precision', default='fp16', choices=['fp16', 'bf16'], help='Mixed precision')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')

    args = parser.parse_args()

    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()

    # ==============================
    # Initialize Booster
    # ==============================
    if args.plugin == 'gemini':
        plugin = GeminiPlugin(precision=args.mixed_precision,
                              placement_policy='auto',
                              initial_scale=2**16,
                              max_norm=args.grad_clip)
    elif args.plugin == 'gemini_cpu':
        plugin = GeminiPlugin(precision=args.mixed_precision,
                              placement_policy='cpu',
                              initial_scale=2**16,
                              max_norm=args.grad_clip)
    elif args.plugin == 'zero2':
        plugin = LowLevelZeroPlugin(stage=2,
                                    precision=args.mixed_precision,
                                    initial_scale=2**16,
                                    max_norm=args.grad_clip)
    elif args.plugin == 'zero2_cpu':
        plugin = LowLevelZeroPlugin(stage=2,
                                    precision=args.mixed_precision,
                                    initial_scale=2**16,
                                    cpu_offload=True,
                                    max_norm=args.grad_clip)
    else:
        raise ValueError(f'Unknown plugin {args.plugin}')

    booster = Booster(plugin=plugin)

    # ==============================
    # Initialize Tokenizer, Dataset and Dataloader
    # ==============================
    tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')
    # follows fast chat: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py#L257
    tokenizer.pad_token = tokenizer.unk_token

    dataset = load_dataset(args.dataset)
    train_ds = dataset['train']
    dataloader = plugin.prepare_dataloader(train_ds,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=partial(tokenize_batch,
                                                              tokenizer=tokenizer,
                                                              max_length=args.max_length))

    # ==============================
    # Initialize Model, Optimizer and LR Scheduler
    # ==============================
    config = MODEL_CONFIGS[args.config]
    init_ctx = LazyInitContext(
        default_device=get_current_device()) if isinstance(plugin, GeminiPlugin) else nullcontext()

    with init_ctx:
        model = LlamaForCausalLM(config)

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    model_numel = get_model_numel(model)
    coordinator.print_on_master(f'Model params: {format_numel_str(model_numel)}')

    optimizer = HybridAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weigth_decay)
    lr_scheduler = CosineAnnealingWarmupLR(optimizer,
                                           total_steps=args.num_epochs * len(dataloader),
                                           warmup_steps=args.warmup_steps,
                                           eta_min=0.1 * args.lr)

    model, optimizer, _, dataloader, lr_scheduler = booster.boost(model,
                                                                  optimizer,
                                                                  dataloader=dataloader,
                                                                  lr_scheduler=lr_scheduler)

    coordinator.print_on_master(f'Booster init max CUDA memory: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB')
    coordinator.print_on_master(
        f'Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB')

    for epoch in range(args.num_epochs):
        with tqdm(dataloader, desc=f'Epoch {epoch}', disable=not coordinator.is_master()) as pbar:
            for batch in pbar:
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs[0]
                booster.backward(loss, optimizer)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                pbar.set_postfix({'loss': loss.item()})

    coordinator.print_on_master(f'Max CUDA memory usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB')


if __name__ == '__main__':
    main()
