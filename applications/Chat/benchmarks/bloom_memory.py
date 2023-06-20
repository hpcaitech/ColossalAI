import argparse
import os
import resource
from contextlib import contextmanager
from copy import deepcopy

import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
from coati.models.base import RewardModel
from coati.models.bloom import BLOOMActor, BLOOMCritic
from coati.trainer import PPOTrainer
from coati.trainer.callbacks import PerformanceEvaluator
from coati.trainer.strategies import ColossalAIStrategy, Strategy, TPZeroStrategy
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.modeling_utils import no_init_weights
from transformers.models.bloom.configuration_bloom import BloomConfig

from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam


def get_model_numel(model: nn.Module, strategy: Strategy) -> int:
    numel = sum(p.numel() for p in model.parameters())
    if isinstance(strategy, ColossalAIStrategy) and strategy.stage == 3 and strategy.shard_init:
        numel *= dist.get_world_size()
    return numel


def get_max_memory() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def get_memory():
    return psutil.Process(os.getpid()).memory_full_info().rss


@contextmanager
def low_precision_init(target_dtype: torch.dtype = torch.float16):
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(target_dtype)
        yield
    finally:
        torch.set_default_dtype(dtype)


def format_model_numel(model_dict: dict) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    outputs = ''
    for name, numel in model_dict.items():
        outputs += f'{name}: '
        if numel >= B:
            outputs += f'{numel / B:.2f} B\n'
        elif numel >= M:
            outputs += f'{numel / M:.2f} M\n'
        elif numel >= K:
            outputs += f'{numel / K:.2f} K\n'
        else:
            outputs += f'{numel}\n'
    return outputs


def get_gpt_config(model_name: str) -> BloomConfig:
    model_map = {
        '350m': BloomConfig(hidden_size=1024, n_layer=24, n_head=16),
        '560m': BloomConfig.from_pretrained('bigscience/bloom-560m'),
        '1.1b': BloomConfig.from_pretrained('bigscience/bloom-1b1'),
        '1.7b': BloomConfig.from_pretrained('bigscience/bloom-1b7'),
        '3b': BloomConfig.from_pretrained('bigscience/bloom-3b'),
        '7b': BloomConfig.from_pretrained('bigscience/bloom-7b1'),
        '66b': BloomConfig(hidden_size=9216, n_layer=64, n_head=72),
        '175b': BloomConfig(hidden_size=12288, n_layer=96, n_head=128),
    }
    try:
        return model_map[model_name]
    except KeyError:
        raise ValueError(f'Unknown model "{model_name}"')


def main(args):
    if args.strategy == 'gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda', initial_scale=2**5)
    elif args.strategy == 'gemini_cpu':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cpu', initial_scale=2**5)
    elif args.strategy == 'gemini_reshard':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda_reshard', initial_scale=2**5)
    elif args.strategy == 'tp_zero2':
        strategy = TPZeroStrategy(args.tp_size, zero_stage=2, initial_scale=2**5)
    elif args.strategy == 'tp_zero2_cpu':
        strategy = TPZeroStrategy(args.tp_size, zero_stage=2, initial_scale=2**5, cpu_offload=True)
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    coordinator = DistCoordinator()
    model_config = get_gpt_config(args.model)
    with strategy.model_init_context(), no_init_weights(), low_precision_init():
        actor = BLOOMActor(config=model_config, lora_rank=args.lora_rank, checkpoint=args.grad_checkpoint)
        actor.model.tie_weights()

    actor_numel = get_model_numel(actor, strategy)
    coordinator.print_on_master(format_model_numel({'Actor': actor_numel}))
    coordinator.print_on_master(f'Mem after lazy init: {get_memory()/1024**3:.2f} GB')
    with low_precision_init():
        if args.init_optim:
            actor_optim = HybridAdam(actor.parameters(), lr=5e-6)
            (actor, actor_optim) = strategy.prepare((actor, actor_optim))
        else:
            actor = strategy.prepare(actor)

    coordinator.print_on_master(f'Mem: {get_memory()/1024**3:.2f} GB')
    coordinator.print_on_master(f'Peak mem: {get_max_memory()/1024**2:.2f} GB')
    coordinator.print_on_master(f'Peak CUDA mem: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='350m')
    parser.add_argument('-s',
                        '--strategy',
                        choices=[
                            'gemini',
                            'gemini_reshard',
                            'gemini_cpu',
                            'tp_zero2',
                            'tp_zero2_cpu',
                        ],
                        default='gemini_reshard')
    parser.add_argument('-t', '--tp_size', type=int, default=1)
    parser.add_argument('-l', '--lora_rank', type=int, default=0)
    parser.add_argument('-g',
                        '--grad_checkpoint',
                        default=False,
                        action='store_true',
                        help='This uses gradient checkpointing, which can save memory and slow down training.')
    parser.add_argument('-o', '--init_optim', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
