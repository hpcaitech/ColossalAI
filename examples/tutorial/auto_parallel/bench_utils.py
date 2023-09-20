import time
from copy import deepcopy
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from colossalai.auto_parallel.checkpoint import CheckpointSolverRotor
from colossalai.fx import metainfo_trace


def bench(
    gm: torch.fx.GraphModule, criterion: torch.nn.Module, data_gen: Callable, num_steps: int = 5
) -> Tuple[int, int]:
    """Benchmarking a given graph module
    Args:
        gm (torch.fx.GraphModule): The graph module to benchmark.
        criterion (torch.nn.Module): Loss function.
        data_gen (Callable): Data generator.
        num_steps (int, optional): Number of test steps. Defaults to 5.
    Returns:
        Tuple[int, int]: peak memory in MB and step time in MS.
    """
    gm.train()
    gm.cuda()
    step_time = float("inf")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cached = torch.cuda.max_memory_allocated(device="cuda")
    try:
        for _ in range(num_steps):
            args, label = data_gen()
            output, loss = None, None

            torch.cuda.synchronize(device="cuda")
            start = time.time()
            output = gm(*args)
            loss = criterion(output, label)
            loss.backward()
            torch.cuda.synchronize(device="cuda")
            step_time = min(step_time, time.time() - start)

            for child in gm.children():
                for param in child.parameters():
                    param.grad = None
            del args, label, output, loss
    except:
        del args, label, output, loss
    gm.to("cpu")
    torch.cuda.empty_cache()
    peak_mem = (torch.cuda.max_memory_allocated(device="cuda") - cached) / 1024**2
    return peak_mem, step_time * 1.0e3


def bench_rotor(
    gm: torch.fx.GraphModule,
    criterion: torch.nn.Module,
    data_gen: Callable,
    num_steps: int = 5,
    sample_points: int = 20,
    free_memory: int = torch.cuda.mem_get_info()[0],
    start_factor: int = 4,
) -> Tuple[np.array, list, list]:
    """Auto Checkpoint Rotor Algorithm benchmarking
    Benchmarks the Auto Checkpoint Rotor Algorithm for a given graph module and data.
    Args:
        gm (torch.fx.GraphModule): The graph module to benchmark.
        criterion (torch.nn.Module): Loss function.
        data_gen (Callable): Data generator.
        num_steps (int, optional): Number of test steps. Defaults to 5.
        sample_points (int, optional): Number of sample points. Defaults to 20.
        free_memory (int, optional): Max memory budget in Byte. Defaults to torch.cuda.mem_get_info()[0].
        start_factor (int, optional): Start memory budget factor for benchmark, the start memory budget
        will be free_memory / start_factor. Defaults to 4.
    Returns:
        Tuple[np.array, list, list]: return budgets vector (MB), peak memory vector (MB), step time vector (MS).
    """
    peak_hist, step_hist = [], []
    raw_graph = deepcopy(gm.graph)
    for budget in np.linspace(free_memory // start_factor, free_memory, sample_points):
        gm = metainfo_trace(gm, *data_gen()[0])
        solver = CheckpointSolverRotor(gm.graph, free_memory=budget)
        try:
            gm.graph = solver.solve(verbose=False)
            peak_memory, step_time = bench(gm, criterion, data_gen, num_steps=num_steps)
        except:
            peak_memory, step_time = budget / 1024**2, float("inf")
        peak_hist.append(peak_memory)
        step_hist.append(step_time)
        gm.graph = deepcopy(raw_graph)
    return np.linspace(free_memory // start_factor, free_memory, sample_points) / 1024**2, peak_hist, step_hist


class GPTLMModel(nn.Module):
    """
    GPT Model
    """

    def __init__(
        self,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        max_seq_len=1024,
        vocab_size=50257,
        checkpoint=False,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(
            GPT2Config(
                n_embd=hidden_size,
                n_layer=num_layers,
                n_head=num_attention_heads,
                n_positions=max_seq_len,
                n_ctx=max_seq_len,
                vocab_size=vocab_size,
            )
        )
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


class GPTLMLoss(nn.Module):
    """
    GPT Loss
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def gpt2_medium(checkpoint=False):
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_xl(checkpoint=False):
    return GPTLMModel(hidden_size=1600, num_layers=48, num_attention_heads=32, checkpoint=checkpoint)


def gpt2_6b(checkpoint=False):
    return GPTLMModel(hidden_size=4096, num_layers=30, num_attention_heads=16, checkpoint=checkpoint)


def data_gen_gpt2(batch_size, seq_len, vocab_size, device="cuda:0"):
    """
    Generate random data for gpt2 benchmarking
    """
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return (input_ids, attention_mask), attention_mask


def data_gen_resnet(batch_size, shape, device="cuda:0"):
    """
    Generate random data for resnet benchmarking
    """
    data = torch.empty(batch_size, *shape, device=device)
    label = torch.empty(batch_size, dtype=torch.long, device=device).random_(1000)
    return (data,), label
