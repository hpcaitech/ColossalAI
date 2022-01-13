import os

import colossalai
import torch
from colossalai.core import global_context as gpc
from colossalai.engine.schedule import InterleavedPipelineSchedule
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import CosineAnnealingWarmupLR
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader, is_using_pp
from model_zoo.gpt import GPTLMLoss

from data import WebtextDataset


def train_gpt():
    disable_existing_loggers()
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    if args.from_torch:
        colossalai.launch_from_torch(config=args.config, seed=42)
    else:
        # standard launch
        colossalai.launch(config=args.config,
                          rank=args.rank,
                          world_size=args.world_size,
                          local_rank=args.local_rank,
                          host=args.host,
                          port=args.port,
                          seed=42)

    logger = get_dist_logger()
    if hasattr(gpc.config, 'LOG_PATH'):
        log_path = gpc.config.LOG_PATH
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logger.log_to_file(log_path)

    train_dataset = WebtextDataset(os.environ['DATA'], seq_len=gpc.config.SEQ_LENGTH)
    train_dataloader = get_dataloader(train_dataset,
                                      seed=42,
                                      batch_size=gpc.config.BATCH_SIZE // gpc.data_parallel_size,
                                      num_workers=1,
                                      pin_memory=True,
                                      shuffle=True,
                                      drop_last=True)
    logger.info(f'Loaded {len(train_dataset)}/{len(train_dataloader)} samples/batches', ranks=[0])

    model = gpc.config.model.pop('type')(**gpc.config.model)
    if is_using_pp():
        schedule = gpc.config.schedule.pop('type')(**gpc.config.schedule)
        # tensor_shape = getattr(gpc.config, 'TENSOR_SHAPE', None)
        # if hasattr(gpc.config, 'NUM_CHUNKS'):
        #     schedule = InterleavedPipelineSchedule(gpc.config.NUM_MICRO_BATCHES,
        #                                            gpc.config.NUM_CHUNKS,
        #                                            tensor_shape=tensor_shape,
        #                                            scatter_gather_tensors=True)
        if isinstance(schedule, InterleavedPipelineSchedule) and not isinstance(model, torch.nn.ModuleList):
            model = torch.nn.ModuleList([model])
        # else:
        #     schedule = PipelineSchedule(gpc.config.NUM_MICRO_BATCHES,
        #                                 tensor_shape=tensor_shape,
        #                                 scatter_gather_tensors=True)
    else:
        schedule = None

    numel = 0
    for p in model.parameters():
        numel += p.numel()
    logger.info(
        f'Rank {gpc.get_global_rank()}: {numel / (1024*1024):.2f} M parameters | memory usage = {torch.cuda.memory_allocated() / (1024 * 1024 * 1024):.2f} GB.'
    )

    criterion = GPTLMLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)

    steps_per_epoch = len(train_dataloader) // gpc.config.gradient_accumulation

    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,
                                           total_steps=gpc.config.NUM_EPOCHS * steps_per_epoch,
                                           warmup_steps=gpc.config.WARMUP_EPOCHS * steps_per_epoch,
                                           eta_min=1e-5)

    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model=model,
                                                                      optimizer=optimizer,
                                                                      criterion=criterion,
                                                                      train_dataloader=train_dataloader,
                                                                      lr_scheduler=lr_scheduler)

    timer = MultiTimer()

    trainer = Trainer(engine=engine, logger=logger, timer=timer, schedule=schedule)

    hook_list = [
        hooks.LogMetricByEpochHook(logger=logger),
        hooks.LogMetricByStepHook(),
        hooks.LossHook(),
        hooks.ThroughputHook(ignored_steps=5),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
        # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
        # hooks.LogMemoryByEpochHook(logger),
        # hooks.LogTimingByEpochHook(timer, logger, ignore_num_train_steps=5),
        # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    logger.info("Training start", ranks=[0])
    torch.cuda.reset_peak_memory_stats()
    trainer.fit(train_dataloader=train_dataloader,
                epochs=gpc.config.NUM_EPOCHS,
                max_steps=10,
                hooks=hook_list,
                return_output_label=False,
                display_progress=True)

    logger.info(
        f'Rank {gpc.get_global_rank()}: peak memory usage = {torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024):.2f} GB.'
    )


if __name__ == '__main__':
    train_gpt()
