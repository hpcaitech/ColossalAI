#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time

import colossalai
import torch
from colossalai import initialize
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_global_dist_logger
from colossalai.utils import empty_cache, print_rank_0, report_memory_usage
from tqdm import tqdm

WAIT_STEPS = 3
WARMUP_STEPS = 50
ACTIVE_STEPS = 100
PROFILE_CYCLE = WAIT_STEPS + WARMUP_STEPS + ACTIVE_STEPS


def _train_epoch(epoch, engine, dataloader, profiler=None):
    logger = get_global_dist_logger()
    print_rank_0('[Epoch %d] training start' % (epoch), logger)
    engine.train()
    data_iter = iter(dataloader)

    train_loss = 0
    batch_cnt = 0
    num_samples = 0
    now = time.time()
    epoch_start = now
    progress = range(PROFILE_CYCLE)
    if gpc.get_global_rank() == 0:
        progress = tqdm(progress, desc='[Epoch %d]' % epoch, miniters=1)
    for step in progress:
        cur_lr = engine.optimizer.param_groups[0]['lr']

        _, targets, loss = engine.step(data_iter)
        if profiler is not None:
            profiler.step()

        batch_size = targets[0].size(0) * engine._grad_accum_size * gpc.data_parallel_size
        train_loss += loss.item()
        num_samples += batch_size
        batch_cnt += 1

        batch_time = time.time() - now
        now = time.time()
        if gpc.get_global_rank() == 0:
            print_features = dict(lr='%g' % cur_lr,
                                  loss='%.3f' % (train_loss / (step + 1)),
                                  throughput='%.3f (images/sec)' % (batch_size / (batch_time + 1e-12)))
            progress.set_postfix(**print_features)

    epoch_end = time.time()
    epoch_loss = train_loss / batch_cnt
    epoch_throughput = num_samples / (epoch_end - epoch_start + 1e-12)
    print_rank_0('[Epoch %d] Loss: %.3f | Throughput: %.3f (samples/sec)' % (epoch, epoch_loss, epoch_throughput),
                 logger)
    if gpc.get_global_rank() == 0:
        report_memory_usage('Memory usage')


def test_cifar():
    engine, train_dataloader, test_dataloader = initialize()

    logger = get_global_dist_logger()
    logger.info("Train start", ranks=[0])
    data_iter = iter(train_dataloader)
    output, targets, loss = engine.step(data_iter)
    if gpc.get_global_rank() == 0:
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=WAIT_STEPS, warmup=WARMUP_STEPS, active=ACTIVE_STEPS),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f'./log_cifar_{gpc.config.parallel.tensor.mode}_{gpc.get_world_size(ParallelMode.GLOBAL)}'),
                record_shapes=True,
                # profile_memory=True,
                with_flops=True,
                with_modules=True,
        ) as prof:
            _train_epoch(0, engine, train_dataloader, prof)

        torch.cuda.synchronize()

        print('Test complete. Generating profiling report ...')
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total"))

        torch.distributed.barrier()
    else:
        _train_epoch(0, engine, train_dataloader)
        torch.cuda.synchronize()
        torch.distributed.barrier()


if __name__ == '__main__':
    test_cifar()
