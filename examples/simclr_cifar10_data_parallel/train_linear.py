from colossalai.nn.metric import Accuracy
import torch
import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import get_dataloader, MultiTimer
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR

from torchvision.datasets import CIFAR10
from myhooks import TotalBatchsizeHook
from models.linear_eval import Linear_eval
from augmentation import LeTransform

def build_dataset_train():
    augment = LeTransform()
    train_dataset = CIFAR10(root=gpc.config.dataset.root, 
                                    transform=augment,
                                    train=True)
                         
    return get_dataloader(
        dataset=train_dataset,
        shuffle=True, 
        num_workers = 1,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
    )

def build_dataset_test():
    augment = LeTransform()
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
    colossalai.launch_from_torch(config='./le_config.py')

    # get logger
    logger = get_dist_logger()

    ## build model
    model = Linear_eval(model='resnet18', class_num=10)

    # build dataloader
    train_dataloader = build_dataset_train()
    test_dataloader = build_dataset_test()

    # build loss
    criterion = torch.nn.CrossEntropyLoss()

    # build optimizer
    optimizer = colossalai.nn.FusedSGD(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY, momentum=gpc.config.MOMENTUM)
    
    # lr_scheduelr
    lr_scheduler = CosineAnnealingWarmupLR(optimizer, warmup_steps=5, total_steps=gpc.config.NUM_EPOCHS)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model, optimizer, criterion, train_dataloader, test_dataloader
    )
    logger.info("initialized colossalai components", ranks=[0])

    ## Load trained self-supervised SimCLR model
    engine.model.load_state_dict(torch.load(f'./ckpt/{gpc.config.LOG_NAME}/epoch{gpc.config.EPOCH}-tp0-pp0.pt')['model'], strict=False)
    logger.info("pretrained model loaded", ranks=[0])

    # build a timer to measure time
    timer = MultiTimer()

    # build trainer
    trainer = Trainer(engine=engine, logger=logger, timer=timer)

    # build hooks
    hook_list = [
        hooks.LossHook(),
        hooks.AccuracyHook(accuracy_func=Accuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LRSchedulerHook(lr_scheduler, by_epoch=True),
        TotalBatchsizeHook(),

        # comment if you do not need to use the hooks below
        hooks.SaveCheckpointHook(interval=5, checkpoint_dir=f'./ckpt/{gpc.config.LOG_NAME}-eval'),
        hooks.TensorboardHook(log_dir=f'./tb_logs/{gpc.config.LOG_NAME}-eval', ranks=[0]),
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