import argparse
import resource
from contextlib import contextmanager, nullcontext

from random import randint
import time


import torch
import torch.nn as nn

from attn import SUPPORT_XFORMERS, replace_xformers

import transformers
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from performance_evaluator import PerformanceEvaluator, Timer
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.modeling_utils import no_init_weights
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, TorchFSDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from colossalai.zero.gemini.placement_policy import AutoPlacementPolicy, ConstPlacementPolicy

MODEL_CONFIGS = {
    '7b': LlamaConfig(),
    '13b': LlamaConfig(hidden_size=5120, intermediate_size=13760, num_hidden_layers=40, num_attention_heads=40),
    '30b': LlamaConfig(hidden_size=6656, intermediate_size=17888, num_hidden_layers=60, num_attention_heads=52),
    '65b': LlamaConfig(hidden_size=8192, intermediate_size=22016, num_hidden_layers=80, num_attention_heads=64),
}

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


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='7b', help='Model configuration')
    parser.add_argument('-p',
                        '--plugin',
                        choices=['gemini', 'gemini_cuda', 'gemini_cpu', 'fsdp', 'fsdp_cpu'],
                        default='gemini',
                        help='Choose which plugin to use')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('-s', '--num_steps', type=int, default=10, help='Number of steps to run')
    parser.add_argument('-i', '--ignore_steps', type=int, default=2, help='Number of steps to ignore')
    parser.add_argument('-g', '--grad_checkpoint', action='store_true', help='Use gradient checkpointing')
    parser.add_argument('-l', '--max_length', type=int, default=2048, help='Max sequence length')
    parser.add_argument('-w', '--warmup_ratio', type=float, default=0.8, help='warm up ratio for auto placement policy')
    parser.add_argument('-m', '--memory_limit', type=int, help='Gemini memory limit in mb')
    parser.add_argument('-x', '--xformers', action='store_true', help='Use xformers')
    args = parser.parse_args()

    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()

    def empty_init():
        pass

    # ==============================
    # Initialize Booster
    # ==============================
    use_empty_init = True
    if args.plugin == 'gemini':
        AutoPlacementPolicy.set_warmup_non_model_data_ratio(args.warmup_ratio)
        plugin = GeminiPlugin(placement_policy='auto')
    elif args.plugin == 'gemini_cuda':
        plugin = GeminiPlugin(placement_policy='cuda')
    elif args.plugin == 'gemini_cpu':
        plugin = GeminiPlugin(placement_policy='cpu')
    elif args.plugin == 'const':
        ConstPlacementPolicy.set_const_memory_boundary(args.memory_limit)
        plugin = GeminiPlugin(placement_policy='const')
    elif args.plugin == 'fsdp':
        if use_empty_init:
            plugin = TorchFSDPPlugin(mixed_precision=MixedPrecision(
                param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16), param_init_fn=empty_init(),)
                # auto_wrap_policy=size_based_auto_wrap_policy)
        else:
            plugin = TorchFSDPPlugin(mixed_precision=MixedPrecision(
                param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16))
    elif args.plugin == 'fsdp_cpu':
        if use_empty_init:
            plugin = TorchFSDPPlugin(mixed_precision=MixedPrecision(param_dtype=torch.float16,
                                                                    reduce_dtype=torch.float16,
                                                                    buffer_dtype=torch.float16),
                                     cpu_offload=CPUOffload(offload_params=True), param_init_fn=empty_init(),)
                                     # auto_wrap_policy=size_based_auto_wrap_policy)
        else:
            plugin = TorchFSDPPlugin(mixed_precision=MixedPrecision(param_dtype=torch.float16,
                                                                    reduce_dtype=torch.float16,
                                                                    buffer_dtype=torch.float16),
                                     cpu_offload=CPUOffload(offload_params=True))
    else:
        raise ValueError(f'Unknown plugin {args.plugin}')

    booster = Booster(plugin=plugin)
    # ==============================
    # Initialize Dataset and Dataloader
    # ==============================

    config = MODEL_CONFIGS[args.config]
    dataset = RandomDataset(num_samples=args.batch_size * args.num_steps * coordinator.world_size,
                            max_length=args.max_length,
                            vocab_size=config.vocab_size)
    dataloader = plugin.prepare_dataloader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # ==============================
    # Initialize Model and Optimizer
    # ==============================
    init_ctx = LazyInitContext(
        default_device=get_current_device()) if isinstance(plugin, GeminiPlugin) else nullcontext()

    timer = Timer()
    timer.start()
    with no_init_weights(), init_ctx:
        model = LlamaForCausalLM(config)
        model.tie_weights()

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    if args.xformers:
        assert SUPPORT_XFORMERS, 'Use flash attention while xfomers is not installed'
        replace_xformers(model)

    model_numel = get_model_numel(model)
    coordinator.print_on_master(f'Model params: {format_numel_str(model_numel)}')
    performance_evaluator = PerformanceEvaluator(model_numel, args.grad_checkpoint, args.ignore_steps)

    optimizer = HybridAdam(model.parameters())
    print("model is on {}".format(model.device))
    model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, dataloader=dataloader, benchmark=use_empty_init)
    timer.end()
    end_time = time.time()
    print("total init time: ", end_time - start_time, "s")
    coordinator.print_on_master(f'Booster init time: {timer.duration:.2f} s')
    coordinator.print_on_master(f'Booster init max CUDA memory: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB')
    coordinator.print_on_master(
        f'Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB')

    for step, batch in enumerate(tqdm(dataloader, desc='Step', disable=not coordinator.is_master())):
        performance_evaluator.on_step_start(step)
        outputs = model(**batch)
        loss = outputs[0]
        booster.backward(loss, optimizer)
        optimizer.step()
        optimizer.zero_grad()
        performance_evaluator.on_step_end(**batch)

    performance_evaluator.on_fit_end()
    coordinator.print_on_master(f'Max CUDA memory usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB')


if __name__ == '__main__':
    main()
