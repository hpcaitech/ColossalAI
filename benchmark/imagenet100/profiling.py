#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
import colossalai

import torch
from tqdm import tqdm

from colossalai import initialize
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_global_dist_logger
from colossalai.utils import print_rank_0, report_memory_usage
from colossalai.utils import empty_cache

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

        batch_size = targets[0].size(
            0) * engine._grad_accum_size * gpc.data_parallel_size
        train_loss += loss.item()
        num_samples += batch_size
        batch_cnt += 1

        batch_time = time.time() - now
        now = time.time()
        if gpc.get_global_rank() == 0:
            print_features = dict(lr='%g' % cur_lr,
                                  loss='%.3f' % (train_loss / (step + 1)),
                                  throughput='%.3f (images/sec)' %
                                  (batch_size / (batch_time + 1e-12)))
            progress.set_postfix(**print_features)

    epoch_end = time.time()
    epoch_loss = train_loss / batch_cnt
    epoch_throughput = num_samples / (epoch_end - epoch_start + 1e-12)
    print_rank_0(
        '[Epoch %d] Loss: %.3f | Throughput: %.3f (samples/sec)' %
        (epoch, epoch_loss, epoch_throughput), logger)
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
                schedule=torch.profiler.schedule(wait=WAIT_STEPS,
                                                 warmup=WARMUP_STEPS,
                                                 active=ACTIVE_STEPS),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f'./log_cifar_{gpc.config.parallel.tensor.mode}_{gpc.get_world_size(ParallelMode.GLOBAL)}'
                ),
                record_shapes=True,
                # profile_memory=True,
                with_flops=True,
                with_modules=True,
        ) as prof:
            _train_epoch(0, engine, train_dataloader, prof)

        torch.cuda.synchronize()

        print('Test complete. Generating profiling report ...')
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total"))

        torch.distributed.barrier()
    else:
        _train_epoch(0, engine, train_dataloader)
        torch.cuda.synchronize()
        torch.distributed.barrier()


def test_imagenet():
    from test_vit_3d import build_dali_train, build_dali_test
    engine, train_dataloader, test_dataloader = initialize(
        train_dataloader=build_dali_train, test_dataloader=build_dali_test)

    logger = get_global_dist_logger()
    logger.info("Train start", ranks=[0])
    if gpc.get_global_rank() == 0:
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=WAIT_STEPS,
                                                 warmup=WARMUP_STEPS,
                                                 active=ACTIVE_STEPS),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f'./log_imagenet_{gpc.config.parallel.tensor.mode}_{gpc.get_world_size(ParallelMode.GLOBAL)}'
                ),
                record_shapes=True,
                # profile_memory=True,
                with_flops=True,
                with_modules=True,
        ) as prof:
            _train_epoch(0, engine, train_dataloader, prof)

        torch.cuda.synchronize()

        print('Test complete. Generating profiling report ...')
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total"))

        torch.distributed.barrier()
    else:
        _train_epoch(0, engine, train_dataloader)
        torch.cuda.synchronize()
        torch.distributed.barrier()


def test_allgather_n_broadcast():
    from colossalai.communication import all_gather
    from colossalai.initialize import init_dist
    from colossalai.utils import get_current_device
    from tqdm import trange

    init_dist()

    logger = get_global_dist_logger()

    BATCH_SIZE = 4024
    HIDDEN_SIZE = 512
    DEPTH = torch.distributed.get_world_size()
    SEQ_LENGTH = 128

    logger.info("Test start", ranks=[0])
    if gpc.get_global_rank() == 0:
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1,
                                                 warmup=5,
                                                 active=10,
                                                 repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f'./log_allgather_n_broadcast_{gpc.get_world_size(ParallelMode.GLOBAL)}'
                ),
                record_shapes=True,
                # profile_memory=True,
                with_flops=True,
                with_modules=True,
        ) as prof:
            tensor_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE // DEPTH)
            for _ in trange(16):
                x = torch.randn(tensor_shape,
                                dtype=torch.float,
                                device=get_current_device())
                x = all_gather(x, -1, ParallelMode.GLOBAL)
                prof.step()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            tensor_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
            for _ in trange(16):
                x = torch.randn(tensor_shape,
                                dtype=torch.float,
                                device=get_current_device())
                x = x.clone()
                torch.distributed.broadcast(x, src=0)
                prof.step()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        print('Test complete. Generating profiling report ...')
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total"))
        torch.distributed.barrier()
    else:
        tensor_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE // DEPTH)
        for _ in range(16):
            x = torch.randn(tensor_shape,
                            dtype=torch.float,
                            device=get_current_device())
            x = all_gather(x, -1, ParallelMode.GLOBAL)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        tensor_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
        for _ in range(16):
            x = torch.randn(tensor_shape,
                            dtype=torch.float,
                            device=get_current_device())
            x = x.clone()
            torch.distributed.broadcast(x, src=0)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.distributed.barrier()


def test_layer():
    from colossalai.initialize import init_dist
    from colossalai.utils import get_current_device
    from tqdm import trange
    from colossalai.nn.layer.parallel_3d import Linear3D, LayerNorm3D

    CONFIG = dict(parallel=dict(pipeline=1, tensor=dict(mode='3d', size=8)),
                  seed=0)

    init_dist(config=CONFIG)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    gpc.set_seed()

    logger = get_global_dist_logger()

    BATCH_SIZE = 512
    HIDDEN_SIZE = 4096
    DEPTH = colossalai.nn.layer.parallel_3d._utils.get_depth_from_env()
    SEQ_LENGTH = 128
    linear1 = Linear3D(HIDDEN_SIZE, HIDDEN_SIZE * 4)
    linear2 = Linear3D(HIDDEN_SIZE * 4, HIDDEN_SIZE)
    dropout = torch.nn.Dropout(0.0)
    norm = LayerNorm3D(HIDDEN_SIZE, eps=1e-5)
    layer = torch.nn.Sequential(linear1, linear2, dropout, norm)

    logger.info("Test start", ranks=[0])
    tensor_shape = (BATCH_SIZE // DEPTH ** 2, SEQ_LENGTH, HIDDEN_SIZE // DEPTH)

    if gpc.get_global_rank() == 0:
        for _ in trange(WARMUP_STEPS):
            x = torch.randn(tensor_shape,
                            dtype=torch.float,
                            device=get_current_device())
            x = layer(x)
            grad = torch.randn(x.shape,
                            dtype=torch.float,
                            device=get_current_device())
            x.backward(grad)
            empty_cache()
        start = time.time()
        for _ in trange(ACTIVE_STEPS):
            x = torch.randn(tensor_shape,
                            dtype=torch.float,
                            device=get_current_device())
            x = layer(x)
            grad = torch.randn(x.shape,
                            dtype=torch.float,
                            device=get_current_device())
            x.backward(grad)
            empty_cache()
        torch.cuda.synchronize()
        end = time.time()
        avg_step_time = (end - start) / ACTIVE_STEPS
        throughput = ACTIVE_STEPS * BATCH_SIZE / (end - start)
        logger.info('Avg step time = {:.3f} s | Throughput = {:.3f} /s'.format(avg_step_time, throughput))
    else:
        for _ in range(WARMUP_STEPS + ACTIVE_STEPS):
            x = torch.randn(tensor_shape,
                            dtype=torch.float,
                            device=get_current_device())
            x = layer(x)
            grad = torch.randn(x.shape,
                            dtype=torch.float,
                            device=get_current_device())
            x.backward(grad)
            empty_cache()
        torch.cuda.synchronize()
    torch.distributed.barrier()

    # if gpc.get_global_rank() == 0:
    #     with torch.profiler.profile(
    #             activities=[
    #                 torch.profiler.ProfilerActivity.CPU,
    #                 torch.profiler.ProfilerActivity.CUDA,
    #             ],
    #             schedule=torch.profiler.schedule(wait=WAIT_STEPS,
    #                                              warmup=WARMUP_STEPS,
    #                                              active=ACTIVE_STEPS),
    #             on_trace_ready=torch.profiler.tensorboard_trace_handler(
    #                 f'./log_layer_3d_{gpc.get_world_size(ParallelMode.GLOBAL)}'
    #             ),
    #             record_shapes=True,
    #             # profile_memory=True,
    #             with_flops=True,
    #             with_modules=True,
    #     ) as prof:
    #         for _ in trange(PROFILE_CYCLE):
    #             x = torch.randn(tensor_shape,
    #                             dtype=torch.float,
    #                             device=get_current_device())
    #             x = layer(x)
    #             grad = torch.randn(x.shape,
    #                             dtype=torch.float,
    #                             device=get_current_device())
    #             x.backward(grad)
    #             prof.step()

    #         torch.cuda.synchronize()

    #     report_memory_usage('Memory usage')
    #     print('Test complete. Generating profiling report ...')
    #     print(
    #         prof.key_averages(group_by_input_shape=True).table(
    #             sort_by="cuda_time_total"))
    #     torch.distributed.barrier()
    # else:
    #     for _ in range(PROFILE_CYCLE):
    #         x = torch.randn(tensor_shape,
    #                         dtype=torch.float,
    #                         device=get_current_device())
    #         x = layer(x)
    #         grad = torch.randn(x.shape,
    #                         dtype=torch.float,
    #                         device=get_current_device())
    #         x.backward(grad)

    #     torch.cuda.synchronize()
    #     torch.distributed.barrier()


if __name__ == '__main__':
    # test_cifar()
    # test_imagenet()
    # test_allgather_n_broadcast()
    test_layer()
