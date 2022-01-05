import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import get_dataloader, MultiTimer
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR

from torchvision.datasets import CIFAR10
from NT_Xentloss import NT_Xentloss
from myhooks import TotalBatchsizeHook
from models.simclr import SimCLR
from augmentation import SimCLRTransform

def build_dataset_train():
    augment = SimCLRTransform()
    train_dataset = CIFAR10(root=gpc.config.dataset.root, 
                                    transform=augment,
                                    train=True,
                                    download=True)
                         
    return get_dataloader(
        dataset=train_dataset,
        shuffle=True, 
        num_workers = 1,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
    )

def build_dataset_test():
    augment = SimCLRTransform()
    val_dataset = CIFAR10(root=gpc.config.dataset.root, 
                                    transform=augment,
                                    train=False)
    
    return get_dataloader(
        dataset=val_dataset,
        add_sampler=False,
        num_workers = 1,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
    )

def main():
    colossalai.launch_from_torch(config='./config.py')
    
    # get logger
    logger = get_dist_logger()

    ## build model
    model = SimCLR(model='resnet18')

    # build dataloader
    train_dataloader = build_dataset_train()
    test_dataloader = build_dataset_test()

    # build loss
    criterion = NT_Xentloss()

    # build optimizer
    optimizer = colossalai.nn.FusedSGD(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY, momentum=gpc.config.MOMENTUM)

    # lr_scheduelr
    lr_scheduler = CosineAnnealingWarmupLR(optimizer, warmup_steps=10, total_steps=gpc.config.NUM_EPOCHS)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model, optimizer, criterion, train_dataloader, test_dataloader
    )
    logger.info("initialized colossalai components", ranks=[0])

    # build a timer to measure time
    timer = MultiTimer()

    # build trainer
    trainer = Trainer(engine=engine, logger=logger, timer=timer)

    # build hooks
    hook_list = [
        hooks.LossHook(),
        hooks.LogMetricByEpochHook(logger),
        hooks.LRSchedulerHook(lr_scheduler, by_epoch=True),
        TotalBatchsizeHook(),

        # comment if you do not need to use the hooks below
        hooks.SaveCheckpointHook(interval=50, checkpoint_dir=f'./ckpt/{gpc.config.LOG_NAME}'),
        hooks.TensorboardHook(log_dir=f'./tb_logs/{gpc.config.LOG_NAME}', ranks=[0]),
    ]

    # start training
    trainer.fit(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        hooks=hook_list,
        display_progress=True,
        test_interval=1
    )


if __name__ == '__main__':
    main()
