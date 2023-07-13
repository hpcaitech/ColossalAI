import argparse
from contextlib import contextmanager

import torch
import deepspeed
import os
import sys
import time
import logging
import numpy as np
import random
import json
import torch
import torch.profiler
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
sys.path.append("..")
from attn import SUPPORT_XFORMERS, replace_xformers
from data_utils import RandomDataset
from model_utils import get_model_numel, format_numel_str, low_precision_init
from tqdm import tqdm

from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
)
from transformers.modeling_utils import no_init_weights

from examples.language.llama.deepspeed_llama.performance_evaluator_ds import PerformanceEvaluator

MODEL_CONFIGS = {
    '7b': LlamaConfig(),
    '65b': LlamaConfig(hidden_size=8192, intermediate_size=22016, num_hidden_layers=80, num_attention_heads=64),
}


def get_current_device() -> torch.device:
    """
    Returns currently selected device (gpu/cpu).
    If cuda available, return gpu, otherwise return cpu.
    """
    if torch.cuda.is_available():
        return torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        return torch.device('cpu')

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('-c', '--config', type=str, default='7b', help='Model configuration')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('-s', '--num_steps', type=int, default=10, help='Number of steps to run')
    parser.add_argument('-i', '--ignore_steps', type=int, default=2, help='Number of steps to ignore')
    parser.add_argument('-g', '--grad_checkpoint', action='store_true', help='Use gradient checkpointing')
    parser.add_argument('-l', '--max_length', type=int, default=2048, help='Max sequence length')
    parser.add_argument('-w', '--world_size', type=int, default=4, help='Distributed world size')
    parser.add_argument('-x', '--xformers', action='store_true', help='Use xformers')
    parser.add_argument('-train_micro_batch_size_per_gpu', type=int, default=2, help='Batch size per GPU')

    return parser


def get_arguments():
    parser = get_argument_parser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    # no cuda mode is not supported
    args.no_cuda = False
    print(args)
    return args


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    start_time = time.time()
    args = get_arguments()
    deepspeed.init_distributed()

    config = MODEL_CONFIGS[args.config]


    # ==============================
    # Initialize Model and Optimizer
    # ==============================

    # with low_precision_init(), no_init_weights():
    with no_init_weights(), deepspeed.zero.Init():
        model = LlamaForCausalLM(config)
        model.tie_weights()

    if args.grad_checkpoint:
        print("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()

    if args.xformers:
        assert SUPPORT_XFORMERS, 'Use flash attention while xfomers is not installed'
        replace_xformers(model)

    model_numel = get_model_numel(model)
    performance_evaluator = PerformanceEvaluator(model_numel, args.grad_checkpoint, args.ignore_steps,
                                                 args.world_size)
    model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                         model=model,
                                                         model_parameters=model.parameters())


    print("loading dataset")
    dataset = RandomDataset(num_samples=args.train_micro_batch_size_per_gpu * args.num_steps * args.world_size,
                            max_length=args.max_length,
                            vocab_size=config.vocab_size)


    data_loader = DataLoader(dataset, batch_size=args.train_micro_batch_size_per_gpu, drop_last=True,
                             sampler=DistributedSampler(dataset, num_replicas=args.world_size, rank=args.local_rank,
                                                        shuffle=True)
                             )

    # ==============================
    # Training
    # ==============================
    for step, batch in enumerate(tqdm(data_loader, desc='Step')):
        print("batch dtype", batch['input_ids'].dtype)
        performance_evaluator.on_step_start(step)
        loss = model_engine(**batch).loss
        #runs backpropagation
        model_engine.backward(loss)

        #weight update
        model_engine.step()
        optimizer.zero_grad()
        performance_evaluator.on_step_end(**batch)

    performance_evaluator.on_fit_end()
    rank = dist.get_rank()
    if rank == 0:
        print(f'Max CUDA memory usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB')

if __name__ == '__main__':
    main()