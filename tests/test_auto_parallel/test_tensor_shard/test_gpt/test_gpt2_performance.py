import copy
import random
from functools import partial
from time import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
import psutil
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import transformers
from torch.fx import GraphModule
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler

from colossalai.auto_parallel.passes.runtime_apply_pass import runtime_apply_pass
from colossalai.auto_parallel.passes.runtime_preparation_pass import runtime_preparation_pass
from colossalai.auto_parallel.tensor_shard.constants import BATCHNORM_MODULE_OP
from colossalai.auto_parallel.tensor_shard.initialize import autoparallelize, initialize_model
from colossalai.auto_parallel.tensor_shard.sharding_strategy import ShardingSpec
from colossalai.auto_parallel.tensor_shard.solver import (
    CostGraph,
    GraphAnalyser,
    Solver,
    SolverOptions,
    StrategiesConstructor,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.initialize import launch, launch_from_torch
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.tensor.shape_consistency import ShapeConsistencyManager, to_global
from colossalai.testing import assert_close, assert_close_loose, parameterize, rerun_if_address_is_in_use
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.utils import free_port
from tests.test_auto_parallel.test_tensor_shard.test_gpt.gpt_modules import GPT2LMHeadModel, GPTLMLoss

BATCH_SIZE = 32
SEQ_LENGTH = 256
HIDDEN_DIM = 16384
NUM_HEADS = 128
NUM_LAYERS = 4
VOCAB_SIZE = 50257
NUM_STEPS = 10
FP16 = True


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def get_tflops(model_numel, batch_size, seq_len, step_time):
    # Tflops_per_GPU = global_batch * global_numel * seq_len * 8 / #gpu
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12) / 4


# Randomly Generated Data
def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def main():
    disable_existing_loggers()
    launch_from_torch(config={})
    logger = get_dist_logger()
    config = transformers.GPT2Config(n_position=SEQ_LENGTH, n_layer=NUM_LAYERS, n_head=NUM_HEADS, n_embd=HIDDEN_DIM)
    if FP16:
        model = GPT2LMHeadModel(config=config).half().to('cuda')
    else:
        model = GPT2LMHeadModel(config=config).to('cuda')
    global_numel = sum([p.numel() for p in model.parameters()])

    meta_input_sample = {
        'input_ids': torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64).to('meta'),
        'attention_mask': torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64).to('meta'),
    }

    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    gm = initialize_model(model, meta_input_sample, device_mesh)

    # build criterion
    criterion = GPTLMLoss()

    optimizer = torch.optim.Adam(gm.parameters(), lr=0.01)
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])
    get_tflops_func = partial(get_tflops, global_numel, BATCH_SIZE, SEQ_LENGTH)
    torch.cuda.synchronize()
    model.train()
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #              schedule=schedule(wait=1, warmup=2, active=2),
    #              on_trace_ready=tensorboard_trace_handler(f'log/dummy_data/bs128_seq128_new'),
    #              record_shapes=True,
    #              profile_memory=True) as prof:
    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    for n in range(10):
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)
        optimizer.zero_grad()
        start = time()
        outputs = gm(input_ids, attn_mask)
        loss = criterion(outputs, input_ids)
        loss.backward()
        optimizer.step()
        # prof.step()
        torch.cuda.synchronize()
        step_time = time() - start
        logger.info(
            f'[{n+1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}',
            ranks=[0])
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    torch.cuda.synchronize()


if __name__ == '__main__':
    main()
