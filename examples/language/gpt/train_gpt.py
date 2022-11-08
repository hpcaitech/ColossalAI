import contextlib
import os

import torch
from dataset.webtext import WebtextDataset
from titans.loss.lm_loss import GPTLMLoss

import colossalai
import colossalai.utils as utils
from colossalai import nn as col_nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import LinearWarmupLR
from colossalai.pipeline.pipelinable import PipelinableContext
from colossalai.trainer import Trainer, hooks
from colossalai.utils import is_using_pp
from colossalai.utils.timer import MultiTimer
from colossalai.zero.init_ctx import ZeroInitContext


def calc_local_model_size(model: torch.nn.Module):
    numel_per_device = 0
    for p in model.parameters():
        numel_per_device += p.numel()
    return numel_per_device


def main():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    disable_existing_loggers()
    if args.from_torch:
        colossalai.launch_from_torch(config=args.config)
    else:
        colossalai.launch_from_slurm(config=args.config, host=args.host, port=29500, seed=42)

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
    use_pipeline = is_using_pp()
    use_interleaved = hasattr(gpc.config.model, 'num_chunks')
    num_chunks = getattr(gpc.config.model, 'num_chunks', 1)
    use_zero3 = hasattr(gpc.config, 'zero')

    if not use_pipeline:
        ctx = contextlib.nullcontext()
        if use_zero3:
            ctx = ZeroInitContext(target_device=torch.cuda.current_device(),
                                  shard_strategy=gpc.config.zero.model_config.shard_strategy,
                                  shard_param=True)
        with ctx:
            model = gpc.config.model.pop('type')(**gpc.config.model)
    else:
        pipelinable = PipelinableContext()
        with pipelinable:
            model = gpc.config.model.pop('type')(**gpc.config.model)

        def mask_function(attention_mask=None):
            if attention_mask is not None:
                batch_size = gpc.config.BATCH_SIZE // gpc.config.NUM_MICRO_BATCHES
                attention_mask = attention_mask.view(batch_size, -1)
                attention_mask = col_nn.partition_batch(attention_mask)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * -10000.0
            return attention_mask

        # GPT2_small exec_seq
        # (lyl)TODO: The exec_seq for gpt3 will be added here and to_layer_list should be more friendly to use.
        exec_seq = ['embed', mask_function, 'blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', (mask_function, "front"), \
                    'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', 'blocks.11', 'norm', 'head']
        pipelinable.to_layer_list(exec_seq)
        ctx = contextlib.nullcontext()
        # (lyl)TODO: Zero context and pipelinable context should be integrated into one context.
        if use_zero3:
            ctx = ZeroInitContext(target_device=torch.cuda.current_device(),
                                  shard_strategy=gpc.config.zero.model_config.shard_strategy,
                                  shard_param=True)
        with ctx:
            model = pipelinable.partition(num_chunks, gpc.pipeline_parallel_size,
                                          gpc.get_local_rank(ParallelMode.PIPELINE))

    if use_zero3:
        numel = ctx.model_numel_tensor.item()
    else:
        numel = calc_local_model_size(model)

    tflop = numel * gpc.config.BATCH_SIZE * gpc.config.SEQ_LEN \
        * gpc.get_world_size(ParallelMode.MODEL) * gpc.get_world_size(ParallelMode.DATA) * 8 / (1024 ** 4)

    criterion = getattr(gpc.config, 'loss_fn', None)
    if criterion is not None:
        criterion = criterion.type()
    else:
        criterion = GPTLMLoss()

    logger.info('Build optimizer', ranks=[0])
    optimizer = gpc.config.optimizer.pop('type')(model.parameters(), **gpc.config.optimizer)

    lr_scheduler = LinearWarmupLR(optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=5)

    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model,
                                                                      optimizer,
                                                                      criterion,
                                                                      train_dataloader=train_dataloader,
                                                                      lr_scheduler=lr_scheduler)
    global_batch_size = gpc.config.BATCH_SIZE * \
        gpc.get_world_size(ParallelMode.DATA) * getattr(gpc.config, "gradient_accumulation", 1)
    logger.info(f'Init done, global batch size = {global_batch_size}', ranks=[0])

    timier = MultiTimer()

    trainer = Trainer(engine=engine, logger=logger, timer=timier)

    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
        hooks.LogMetricByEpochHook(logger),
        hooks.ThroughputHook(ignored_steps=10, tflop_per_step=tflop),
        hooks.LogMetricByStepHook(),
        hooks.LogMemoryByEpochHook(logger),
    ]

    trainer.fit(train_dataloader=train_dataloader,
                epochs=gpc.config.NUM_EPOCHS,
                test_interval=1,
                hooks=hook_list,
                display_progress=True,
                return_output_label=False)


if __name__ == '__main__':
    main()
