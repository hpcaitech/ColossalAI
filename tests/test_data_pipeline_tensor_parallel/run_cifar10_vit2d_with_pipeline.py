import pytest
from pathlib import Path
from colossalai.amp.amp_type import AMP_TYPE
from colossalai.context.parallel_mode import ParallelMode
from colossalai.logging import get_dist_logger
import colossalai
import torch
import os
from colossalai.builder import PipelineModelInitializer
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader, MultiTimer
from colossalai.nn.loss import CrossEntropyLoss2D
from colossalai.trainer.metric import Accuracy2D
from colossalai.trainer import metric, hooks, Trainer
from colossalai.utils.gradient_accumulation import GradAccumLrSchedulerByStep
from colossalai.engine.schedule import PipelineSchedule
from torchvision import transforms
from torchvision.datasets import CIFAR10
from colossalai.nn import LinearWarmupLR
from tqdm import tqdm
import vit_t_2d

BATCH_SIZE = 16
NUM_EPOCHS = 60
WARMUP_EPOCHS = 5
CONFIG = dict(
    parallel=dict(
        pipeline=2,
        tensor=dict(size=4, mode='2d')
    ),
    fp16=dict(
        mode=AMP_TYPE.TORCH
    ),
    gradient_accumulation=2
)


@pytest.mark.dist
@pytest.mark.skip("This test requires more than 8 GPUs, you should invoke this test script using test.sh provided manually")
def test_hybrid_parallel():
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    colossalai.launch_from_slurm(config=CONFIG,
                                 host=args.host,
                                 port=29500)

    logger = get_dist_logger()
    # if gpc.get_global_rank() == 0:
    #     logger.log_to_file('./logs/cifar10_2d_vit',
    #                        suffix='cifar10_2d_vit_ddp1_torch_amp_grad_accum_2_clip_grad_1', mode='w')

    # build vit-t-32
    initializer = PipelineModelInitializer(vit_t_2d.model_cfg, num_chunks=1)
    model = initializer.initialize()

    # build dataloaders
    train_dataset = CIFAR10(
        root=Path(os.environ['DATA']),
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                    0.2023, 0.1994, 0.2010]),
            ]
        )
    )

    test_dataset = CIFAR10(
        root=Path(os.environ['DATA']),
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                    0.2023, 0.1994, 0.2010]),
            ]
        )
    )

    train_dataloader = get_dataloader(dataset=train_dataset,
                                      shuffle=True,
                                      add_sampler=True,
                                      batch_size=BATCH_SIZE,
                                      num_workers=1,
                                      pin_memory=True,
                                      )

    test_dataloader = get_dataloader(dataset=test_dataset,
                                     add_sampler=True,
                                     batch_size=BATCH_SIZE,
                                     num_workers=1,
                                     pin_memory=True,
                                     )

    # build criterion
    criterion = CrossEntropyLoss2D()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    # lr_scheduler
    steps_per_epoch = GradAccumLrSchedulerByStep.compute_effective_steps_per_epoch(train_dataloader, accumulate_size=2)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS
    lr_scheduler = LinearWarmupLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(
        model, optimizer, criterion, train_dataloader, test_dataloader, lr_scheduler)

    timer = MultiTimer()

    schedule = PipelineSchedule(num_microbatches=4)

    trainer = Trainer(
        engine=engine,
        timer=timer,
        logger=logger,
        schedule=schedule
    )

    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
        hooks.Accuracy2DHook(),
        hooks.LogMetricByEpochHook(logger),
    ]

    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=NUM_EPOCHS,
        test_dataloader=test_dataloader,
        test_interval=1,
        hooks=hook_list,
        display_progress=True
    )


if __name__ == '__main__':
    main()
