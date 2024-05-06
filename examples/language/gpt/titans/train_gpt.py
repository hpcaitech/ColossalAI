import argparse
import contextlib
import os

import torch
import torch.nn as nn
from dataset.webtext import WebtextDataset
from titans.model.gpt import GPTLMLoss

import colossalai
import colossalai.utils as utils
from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.trainer import Trainer, hooks
from colossalai.legacy.zero.init_ctx import ZeroInitContext
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import LinearWarmupLR
from colossalai.utils import is_using_pp
from colossalai.utils.timer import MultiTimer


def calc_local_model_size(model: torch.nn.Module):
    numel_per_device = 0
    for p in model.parameters():
        numel_per_device += p.numel()
    return numel_per_device


VOCAB_SIZE = 50257


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_torch", default=False, action="store_true")
    parser.add_argument("--use_dummy_dataset", default=False, action="store_true")
    args = parser.parse_args()
    disable_existing_loggers()
    if args.from_torch:
        colossalai.launch_from_torch()
    else:
        colossalai.launch_from_slurm(host=args.host, port=29500, seed=42)
    logger = get_dist_logger()

    data_path = None if args.use_dummy_dataset else os.environ["DATA"]
    logger.info(f"Build data loader from path {data_path}", ranks=[0])

    train_ds = WebtextDataset(path=data_path, seq_len=gpc.config.SEQ_LEN)
    train_dataloader = utils.get_dataloader(
        train_ds, seed=42, batch_size=gpc.config.BATCH_SIZE, pin_memory=True, shuffle=True, drop_last=True
    )

    logger.info("Build model", ranks=[0])
    use_pipeline = is_using_pp()
    use_interleaved = hasattr(gpc.config.model, "num_chunks")
    use_zero3 = hasattr(gpc.config, "zero")
    ctx = contextlib.nullcontext()
    if use_zero3:
        ctx = ZeroInitContext(
            target_device=torch.cuda.current_device(),
            shard_strategy=gpc.config.zero.model_config.shard_strategy,
            shard_param=True,
        )
    with ctx:
        model = gpc.config.model.pop("type")(**gpc.config.model)
    if use_pipeline and use_interleaved and not isinstance(model, nn.ModuleList):
        model = nn.ModuleList([model])

    if use_zero3:
        numel = ctx.model_numel_tensor.item()
    else:
        numel = calc_local_model_size(model)

    tflop = (
        numel
        * gpc.config.BATCH_SIZE
        * gpc.config.SEQ_LEN
        * gpc.get_world_size(ParallelMode.MODEL)
        * gpc.get_world_size(ParallelMode.DATA)
        * 8
        / (1024**4)
    )

    criterion = getattr(gpc.config, "loss_fn", None)
    if criterion is not None:
        criterion = criterion.type()
    else:
        criterion = GPTLMLoss()
    logger.info("Build optimizer", ranks=[0])
    optimizer = gpc.config.optimizer.pop("type")(model.parameters(), **gpc.config.optimizer)
    lr_scheduler = LinearWarmupLR(optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=5)
    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(
        model, optimizer, criterion, train_dataloader=train_dataloader, lr_scheduler=lr_scheduler
    )
    global_batch_size = (
        gpc.config.BATCH_SIZE * gpc.get_world_size(ParallelMode.DATA) * getattr(gpc.config, "gradient_accumulation", 1)
    )
    logger.info(f"Init done, global batch size = {global_batch_size}", ranks=[0])
    timier = MultiTimer()
    trainer = Trainer(engine=engine, logger=logger, timer=timier)
    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
        hooks.LogMetricByEpochHook(logger),
        hooks.ThroughputHook(ignored_steps=10, tflop_per_step=tflop),
        hooks.LogMetricByStepHook(),
        hooks.LogMemoryByEpochHook(logger),
        # hooks.LogMemoryByEpochHook(logger),
        # hooks.LogTimingByEpochHook(timer, logger),
    ]
    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        test_interval=1,
        hooks=hook_list,
        display_progress=True,
        return_output_label=False,
    )


if __name__ == "__main__":
    main()
