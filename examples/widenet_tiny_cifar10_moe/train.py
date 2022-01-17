import os

import colossalai
import torch
import torchvision
from torchvision import transforms
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import Accuracy
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.trainer import Trainer
from colossalai.trainer.hooks import (AccuracyHook, LogMemoryByEpochHook,
                                      LogMetricByEpochHook,
                                      LogMetricByStepHook,
                                      LogTimingByEpochHook, LossHook,
                                      LRSchedulerHook, ThroughputHook)
from colossalai.utils import MultiTimer, get_dataloader
from colossalai.nn.loss import MoeCrossEntropyLoss
from model_zoo.moe.models import Widenet
from colossalai.context.random import moe_set_seed

DATASET_PATH = str(os.environ['DATA'])  # The directory of your dataset


def build_cifar(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=DATASET_PATH,
                                                 train=True,
                                                 download=True,
                                                 transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, transform=transform_test)
    train_dataloader = get_dataloader(dataset=train_dataset,
                                      shuffle=True,
                                      batch_size=batch_size,
                                      num_workers=4,
                                      pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_dataloader, test_dataloader


def train_cifar():
    args = colossalai.get_default_parser().parse_args()
    colossalai.launch_from_torch(config=args.config)

    logger = get_dist_logger()
    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    moe_set_seed(42)
    model = Widenet(
        num_experts=4,
        capacity_factor=1.2,
        img_size=32,
        patch_size=4,
        num_classes=10,
        depth=6,
        d_model=512,
        num_heads=2,
        d_kv=128,
        d_ff=2048
    )

    train_dataloader, test_dataloader = build_cifar(gpc.config.BATCH_SIZE // gpc.data_parallel_size)
    criterion = MoeCrossEntropyLoss(aux_weight=0.01, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=gpc.config.LEARNING_RATE,
                                  weight_decay=gpc.config.WEIGHT_DECAY)

    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,
                                           total_steps=gpc.config.NUM_EPOCHS,
                                           warmup_steps=gpc.config.WARMUP_EPOCHS)

    engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(model=model,
                                                                                    optimizer=optimizer,
                                                                                    criterion=criterion,
                                                                                    train_dataloader=train_dataloader,
                                                                                    test_dataloader=test_dataloader,
                                                                                    lr_scheduler=lr_scheduler)

    logger.info("Engine is built", ranks=[0])

    timer = MultiTimer()
    trainer = Trainer(engine=engine, logger=logger, timer=timer)
    logger.info("Trainer is built", ranks=[0])

    hooks = [
        LogMetricByEpochHook(logger=logger),
        LogMetricByStepHook(),
        AccuracyHook(accuracy_func=Accuracy()),
        LossHook(),
        ThroughputHook(),
        LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True)
    ]

    logger.info("Train start", ranks=[0])
    trainer.fit(train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                epochs=gpc.config.NUM_EPOCHS,
                hooks=hooks,
                display_progress=True,
                test_interval=1)


if __name__ == '__main__':
    train_cifar()
