from colossalai.logging import get_dist_logger, disable_existing_loggers
import colossalai
import os
from colossalai.core import global_context as gpc
from colossalai.utils.timer import MultiTimer
from colossalai.zero import zero3_model_context
import colossalai.utils as utils
from colossalai.trainer import hooks, Trainer
from colossalai.nn import LinearWarmupLR
from colossalai.context import ParallelMode
import torch.nn as nn
import torch
from dataset.webtext import WebtextDataset
import contextlib
from colossalai.engine.schedule import PipelineSchedule, InterleavedPipelineSchedule, NonPipelineSchedule
from model.gpt import GPTLMLoss

# profiler setup
# WAIT_STEPS = 1
# WARMUP_STEPS = 1
# ACTIVE_STEPS = 3

def main():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    disable_existing_loggers()
    if args.from_torch:
        colossalai.launch_from_torch(config=args.config)
    else:
        colossalai.launch_from_slurm(config=args.config,
                                     host=args.host,
                                     port=29500,
                                     seed=42)

    logger = get_dist_logger()

    logger.info('Build data loader', ranks=[0])
    train_ds = WebtextDataset(os.environ['DATA'], seq_len=gpc.config.SEQ_LEN)
    train_dataloader = utils.get_dataloader(train_ds,
                                            seed=42,
                                            batch_size=gpc.config.BATCH_SIZE,
                                            pin_memory=True,
                                            shuffle=True,
                                            drop_last=True)

    logger.info('Build model', ranks=[0])
    use_pipeline = getattr(gpc.config.parallel, 'pipeline', None)
    use_interleaved = hasattr(gpc.config.model, 'num_chunks')
    use_zero3 = hasattr(gpc.config, 'zero') and gpc.config.zero.level == 3
    ctx = zero3_model_context() if use_zero3 else contextlib.nullcontext()
    with ctx:
        model = gpc.config.model.pop('type')(**gpc.config.model)
    if use_interleaved and not isinstance(model, nn.ModuleList):
        model = nn.ModuleList([model])

    criterion = getattr(gpc.config, 'loss_fn', None)
    if criterion is not None:
        criterion = criterion.type()
    else:
        criterion = GPTLMLoss()

    logger.info('Build optimizer', ranks=[0])
    optimizer = gpc.config.optimizer.pop('type')(
        model.parameters(), **gpc.config.optimizer)

    lr_scheduler = LinearWarmupLR(
        optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=5)

    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model,
                                                                      optimizer,
                                                                      criterion,
                                                                      train_dataloader=train_dataloader,
                                                                      lr_scheduler=lr_scheduler)
    logger.info('Init done', ranks=[0])
    tensor_shape = getattr(gpc.config, 'TENSOR_SHAPE', None)
    if use_pipeline is not None and (use_pipeline != 1) :
        if use_interleaved:
            logger.info('Build InterleavedPipelineSchedule', ranks=[0])
            schedule = InterleavedPipelineSchedule(gpc.config.NUM_MICRO_BATCHES,
                                                gpc.config.model.num_chunks, tensor_shape=tensor_shape, scatter_gather_tensors=True)
        else:
            logger.info('Build PipelineSchedule', ranks=[0])
            schedule = PipelineSchedule(gpc.config.NUM_MICRO_BATCHES,
                                        tensor_shape=tensor_shape, scatter_gather_tensors=True)
    else:
        logger.info('Build InterleavedPipelineSchedule', ranks=[0])
        schedule = NonPipelineSchedule()
    

    timier = MultiTimer()

    trainer = Trainer(
        engine=engine,
        logger=logger,
        schedule=schedule,
        timer=timier
    )

    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
        hooks.LogMetricByEpochHook(logger),
        hooks.ThroughputHook(),
        hooks.LogMetricByStepHook(),
        # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
        # hooks.LogMemoryByEpochHook(logger),
        # hooks.LogTimingByEpochHook(timer, logger),
        # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        test_interval=1,
        hooks=hook_list,
        display_progress=True,
        return_output_label=False
    )

    # add profiler
    # if gpc.get_global_rank() == 0:
    #     with torch.profiler.profile(
    #             activities=[
    #                 # torch.profiler.ProfilerActivity.CPU,
    #                 torch.profiler.ProfilerActivity.CUDA,
    #             ],
    #             schedule=torch.profiler.schedule(wait=WAIT_STEPS,
    #                                              warmup=WARMUP_STEPS,
    #                                              active=ACTIVE_STEPS),
    #             on_trace_ready=torch.profiler.tensorboard_trace_handler(
    #                 f'./log_gpt_{gpc.config.parallel.tensor.mode}_{gpc.get_world_size(ParallelMode.GLOBAL)}'
    #             ),
    #             record_shapes=True,
    #             profile_memory=True,
    #             # with_flops=True,
    #             # with_modules=True,
    #     ) as prof:
    #         if prof is not None:
    #             print('profiler is ok before entering trainer')
    #         trainer.fit(
    #             train_dataloader=train_dataloader,
    #             epochs=gpc.config.NUM_EPOCHS,
    #             test_interval=1,
    #             hooks=hook_list,
    #             display_progress=True,
    #             return_output_label=False,
    #             profiler=prof,
    #             max_steps=WAIT_STEPS + WARMUP_STEPS + ACTIVE_STEPS,
    #         )

    #     torch.cuda.synchronize()

    #     print('Test complete. Generating profiling report ...')

    #     torch.distributed.barrier()
    # else:
    #     trainer.fit(
    #             train_dataloader=train_dataloader,
    #             epochs=gpc.config.NUM_EPOCHS,
    #             test_interval=1,
    #             hooks=hook_list,
    #             display_progress=True,
    #             return_output_label=False,
    #             max_steps=WAIT_STEPS + WARMUP_STEPS + ACTIVE_STEPS,
    #         )
    #     torch.cuda.synchronize()
    #     torch.distributed.barrier()

if __name__ == '__main__':
    main()
