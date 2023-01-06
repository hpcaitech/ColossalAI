from functools import partial
from time import time
from typing import Dict, Optional, Tuple, Union

import psutil
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import transformers
from gpt_modules import GPT2LMHeadModel, GPTLMLoss
from torch.fx import GraphModule

from colossalai.auto_parallel.tensor_shard.initialize import autoparallelize, initialize_model
from colossalai.core import global_context as gpc
from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch_from_torch
from colossalai.logging import disable_existing_loggers, get_dist_logger

BATCH_SIZE = 8
SEQ_LENGTH = 128
HIDDEN_DIM = 3072
NUM_HEADS = 16
NUM_LAYERS = 1
VOCAB_SIZE = 50257
NUM_STEPS = 10
FP16 = False


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

    # Both device mesh initialization and model initialization will be integrated into autoparallelize
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    # Enable auto-parallel
    gm, solution = initialize_model(model, meta_input_sample, device_mesh, return_solution=True)

    # print solution on rank 0
    if gpc.get_global_rank() == 0:
        for node_strategy in solution:
            print(node_strategy)

    # build criterion
    criterion = GPTLMLoss()

    optimizer = torch.optim.Adam(gm.parameters(), lr=0.01)
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])
    get_tflops_func = partial(get_tflops, global_numel, BATCH_SIZE, SEQ_LENGTH)
    torch.cuda.synchronize()
    model.train()

    for n in range(10):
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)
        optimizer.zero_grad()
        start = time()
        outputs = gm(input_ids, attn_mask)
        loss = criterion(outputs, input_ids)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_time = time() - start
        logger.info(
            f'[{n+1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}',
            ranks=[0])
    torch.cuda.synchronize()


if __name__ == '__main__':
    main()
