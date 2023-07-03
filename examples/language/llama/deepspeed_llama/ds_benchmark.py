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
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    LlamaTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_utils import no_init_weights
from transformers.utils.versions import require_version

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

def get_model_numel(model: torch.nn.Module) -> int:
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

def get_argument_parser():
    parser = argparse.ArgumentParser()

    # Required_parameter
    # parser.add_argument(
    #     "--config-file",
    #     "--cf",
    #     help="pointer to the configuration file of the experiment",
    #     type=str,
    #     required=True)

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
    parser.add_argument('-train_micro_batch_size_per_gpu', type=int, default=2, help='Batch size per GPU')

    return parser
# ==============================
# random dataloader for llama finetuning
# ==============================


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


@contextmanager
def low_precision_init(target_dtype: torch.dtype = torch.float16):
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(target_dtype)
        yield
    finally:
        torch.set_default_dtype(dtype)



def get_dataloader(args, dataset: Dataset, eval_set=False):
    # if args.local_rank == -1:
    #     train_sampler = RandomSampler(dataset)
    # else:
    train_sampler = DistributedSampler(dataset)
    return (x for x in
            DataLoader(dataset,
                       args.train_micro_batch_size_per_gpu,
                       sampler=train_sampler,
                       num_workers=10))

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
    print(os.environ)
    start_time = time.time()
    args = get_arguments()
    deepspeed.init_distributed()

    config = MODEL_CONFIGS[args.config]


    # ==============================
    # Initialize Model and Optimizer
    # ==============================

    # with low_precision_init(), no_init_weights():
    with low_precision_init(), no_init_weights(), deepspeed.zero.Init():
        model = LlamaForCausalLM(config)
        model.tie_weights()

    if args.grad_checkpoint:
        print("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    model_numel = get_model_numel(model)
    performance_evaluator = PerformanceEvaluator(model_numel, args.grad_checkpoint, args.ignore_steps,
                                                 args.world_size)
    model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                         model=model,
                                                         model_parameters=model.parameters())


    # Set DeepSpeed info
    # args.local_rank = model.network.local_rank
    # args.device = model.network.device
    # model.set_device(args.device)
    # args.fp16 = model.network.fp16_enabled()
    print("loading dataset")
    dataset = RandomDataset(num_samples=args.train_micro_batch_size_per_gpu * args.num_steps * args.world_size,
                            max_length=args.max_length,
                            vocab_size=config.vocab_size)

    def seed_worker(worker_id):
        worker_seed = 1024
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    # data_loader = get_dataloader(args, dataset)
    data_loader = DataLoader(dataset, batch_size=args.train_micro_batch_size_per_gpu, drop_last=True,
                             sampler=DistributedSampler(dataset, num_replicas=args.world_size, rank=args.local_rank,
                                                        shuffle=True)
                             )

    end_time = time.time()
    print(f'Initialization took {end_time - start_time:.2f} seconds')
    for step, batch in enumerate(tqdm(data_loader, desc='Step')):
        #forward() method
        # print(batch)
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