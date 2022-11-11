import os
import colossalai
import torch

from tqdm import tqdm
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import CrossEntropyLoss
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.utils import is_using_pp, get_dataloader
from colossalai.pipeline.pipelinable import PipelinableContext
from titans.model.vit.vit import _create_vit_model
from titans.dataloader.cifar10 import build_cifar


def main():
    # initialize distributed setting
    parser = colossalai.get_default_parser()
    args = parser.parse_args()

    # launch from torch
    colossalai.launch_from_torch(config=args.config)

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    use_pipeline = is_using_pp()

    # create model
    model_kwargs = dict(img_size=gpc.config.IMG_SIZE,
                        patch_size=gpc.config.PATCH_SIZE,
                        hidden_size=gpc.config.HIDDEN_SIZE,
                        depth=gpc.config.DEPTH,
                        num_heads=gpc.config.NUM_HEADS,
                        mlp_ratio=gpc.config.MLP_RATIO,
                        num_classes=10,
                        init_method='jax',
                        checkpoint=gpc.config.CHECKPOINT)

    if use_pipeline:
        pipelinable = PipelinableContext()
        with pipelinable:
            model = _create_vit_model(**model_kwargs)
        pipelinable.to_layer_list()
        pipelinable.policy = "uniform"
        model = pipelinable.partition(
            1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))
    else:
        model = _create_vit_model(**model_kwargs)

    # count number of parameters
    total_numel = 0
    for p in model.parameters():
        total_numel += p.numel()
    if not gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_stage = 0
    else:
        pipeline_stage = gpc.get_local_rank(ParallelMode.PIPELINE)
    logger.info(
        f"number of parameters: {total_numel} on pipeline stage {pipeline_stage}")

    # create dataloaders
    root = os.environ.get('DATA', '../data/cifar10')
    train_dataloader, test_dataloader = build_cifar(
        gpc.config.BATCH_SIZE, root, pad_if_needed=True)

    # create loss function
    criterion = CrossEntropyLoss(label_smoothing=0.1)

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(
    ), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)

    # create lr scheduler
    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,
                                           total_steps=gpc.config.NUM_EPOCHS,
                                           warmup_steps=gpc.config.WARMUP_EPOCHS)

    # initialize
    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model=model,
                                                                         optimizer=optimizer,
                                                                         criterion=criterion,
                                                                         train_dataloader=train_dataloader,
                                                                         test_dataloader=test_dataloader)

    logger.info("Engine is built", ranks=[0])

    data_iter = iter(train_dataloader)

    for epoch in range(gpc.config.NUM_EPOCHS):
        # training
        engine.train()

        if gpc.get_global_rank() == 0:
            description = 'Epoch {} / {}'.format(epoch, gpc.config.NUM_EPOCHS)
            progress = tqdm(range(len(train_dataloader)), desc=description)
        else:
            progress = range(len(train_dataloader))
        for _ in progress:
            engine.zero_grad()
            engine.execute_schedule(data_iter, return_output_label=False)
            engine.step()
            lr_scheduler.step()


if __name__ == '__main__':
    main()
