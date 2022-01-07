
import torch
import os
from torch.utils.data import DataLoader

from dataset.webtext import WebtextDataset
import colossalai
from colossalai.amp import AMP_TYPE
from colossalai.logging import get_dist_logger, disable_existing_loggers
from colossalai.trainer import Trainer, hooks
from colossalai.utils import get_dataloader
from model.gpt import GPTLMLoss
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.zero import zero3_model_context
from colossalai.nn import LinearWarmupLR
from colossalai.engine.schedule import PipelineSchedule, InterleavedPipelineSchedule
from colossalai.utils import MultiTimer
import model as q
import contextlib

from env import set_env

# launch without config file
# BATCH_SIZE = 4
# NUM_EPOCHS = 60

# CONFIG = dict(
#     parallel=dict(
#         pipeline=dict(size=1),
#         tensor=dict(size=4, mode='1d'),
#     ),
#     # fp16=dict(
#     #     mode=AMP_TYPE.NAIVE,
#     #     clip_grad=1
#     # ),
#     gradient_accumulation=2,
# )


def run_trainer():

    disable_existing_loggers()
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    if args.from_torch:
        colossalai.launch_from_torch(config=args.config,
                                     host=os.environ['MASTER_ADDR'],
                                     port=os.environ['MASTER_PORT'])
    else:
        colossalai.launch_from_slurm(config=args.config,
                                     host=args.host,
                                     port=29500,
                                     seed=42)
    logger = get_dist_logger()
    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)
    # instantiate your compoentns
    # model = GPT2_exlarge_1D(True)
    # model = GPT2_medium_1D(True)
    use_zero3 = hasattr(gpc.config, 'zero') and gpc.config.zero.level == 3
    ctx = zero3_model_context() if use_zero3 else contextlib.nullcontext()
    with ctx:
        model = gpc.config.model.pop('type')(**gpc.config.model)

    criterion = q.vocab_parallel_cross_entropy()

    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = gpc.config.optimizer.pop('type')(
        model.parameters(), **gpc.config.optimizer)

    lr_scheduler = LinearWarmupLR(
        optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=5)

    logger.info('Build data loader', ranks=[0])

    train_ds = WebtextDataset('/project/scratch/p200012/dataset/openwebtext/', seq_len=gpc.config.SEQ_LEN)
    # train_ds = WebtextDataset(os.environ['DATA'], seq_len=gpc.config.SEQ_LEN)
    train_dataloader = get_dataloader(train_ds,
                                      seed=42,
                                      batch_size=gpc.config.BATCH_SIZE,
                                      pin_memory=True,
                                      shuffle=True,
                                      drop_last=True)

    logger.info('Build model', ranks=[0])

    logger.info("components are built")
    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model,
                                                                      optimizer,
                                                                      criterion,
                                                                      train_dataloader=train_dataloader,
                                                                      lr_scheduler=lr_scheduler)

    num_model_chunks = getattr(gpc.config.model, 'num_chunks', 1)
    tensor_shape = getattr(gpc.config, 'TENSOR_SHAPE', None)
    if num_model_chunks > 1:
        logger.info('Build InterleavedPipelineSchedule', ranks=[0])
        schedule = InterleavedPipelineSchedule(gpc.config.NUM_MICRO_BATCHES,
                                               num_model_chunks, tensor_shape=tensor_shape)
    else:
        logger.info('Build PipelineSchedule', ranks=[0])
        schedule = PipelineSchedule(gpc.config.NUM_MICRO_BATCHES, tensor_shape=tensor_shape)

    timer = MultiTimer()

    trainer = Trainer(
        engine=engine,
        logger=logger,
        timer=timer,
        schedule=schedule
    )

    hook_list = [
        hooks.LossHook(),
        # hooks.AccuracyHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
        # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
        hooks.LogMetricByEpochHook(logger),
        hooks.LogTimingByEpochHook(timer=timer, logger=logger, ignore_num_train_steps=2)
        # hooks.LogMemoryByEpochHook(logger),
        # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        hooks=hook_list,
        display_progress=True,
        return_output_label=False,
        max_steps=12,
    )


if __name__ == '__main__':
    set_env()
    run_trainer()
