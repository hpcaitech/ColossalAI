import glob
from math import log
import os
import colossalai
from colossalai.nn.metric import Accuracy
import torch

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.nn.lr_scheduler import LinearWarmupLR
from dataloader.imagenet_dali_dataloader import DaliDataloader
from mixup import MixupLoss, MixupAccuracy
from timm.models import vit_base_patch16_224
from myhooks import TotalBatchsizeHook


def build_dali_train():
    root = gpc.config.dali.root
    train_pat = os.path.join(root, 'train/*')
    train_idx_pat = os.path.join(root, 'idx_files/train/*')
    return DaliDataloader(
        sorted(glob.glob(train_pat)),
        sorted(glob.glob(train_idx_pat)),
        batch_size=gpc.config.BATCH_SIZE,
        shard_id=gpc.get_local_rank(ParallelMode.DATA),
        num_shards=gpc.get_world_size(ParallelMode.DATA),
        training=True,
        gpu_aug=gpc.config.dali.gpu_aug,
        cuda=True,
        mixup_alpha=gpc.config.dali.mixup_alpha
    )


def build_dali_test():
    root = gpc.config.dali.root
    val_pat = os.path.join(root, 'validation/*')
    val_idx_pat = os.path.join(root, 'idx_files/validation/*')
    return DaliDataloader(
        sorted(glob.glob(val_pat)),
        sorted(glob.glob(val_idx_pat)),
        batch_size=gpc.config.BATCH_SIZE,
        shard_id=gpc.get_local_rank(ParallelMode.DATA),
        num_shards=gpc.get_world_size(ParallelMode.DATA),
        training=False,
        # gpu_aug=gpc.config.dali.gpu_aug,
        gpu_aug=False,
        cuda=True,
        mixup_alpha=gpc.config.dali.mixup_alpha
    )


def main():
    # initialize distributed setting
    parser = colossalai.get_default_parser()
    args = parser.parse_args()

    # launch from slurm batch job
    colossalai.launch_from_slurm(config=args.config,
                                 host=args.host,
                                 port=args.port,
                                 backend=args.backend
                                 )
    # launch from torch
    # colossalai.launch_from_torch(config=args.config)

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    # build model
    model = vit_base_patch16_224(drop_rate=0.1)

    # build dataloader
    train_dataloader = build_dali_train()
    test_dataloader = build_dali_test()

    # build optimizer
    optimizer = colossalai.nn.Lamb(model.parameters(), lr=1.8e-2, weight_decay=0.1)

    # build loss
    criterion = MixupLoss(loss_fn_cls=torch.nn.CrossEntropyLoss)

    # lr_scheduelr
    lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=50, total_steps=gpc.config.NUM_EPOCHS)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model, optimizer, criterion, train_dataloader, test_dataloader
    )
    logger.info("initialized colossalai components", ranks=[0])

    # build trainer
    trainer = Trainer(engine=engine, logger=logger)

    # build hooks
    hook_list = [
        hooks.LossHook(),
        hooks.AccuracyHook(accuracy_func=MixupAccuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LRSchedulerHook(lr_scheduler, by_epoch=True),
        TotalBatchsizeHook(),

        # comment if you do not need to use the hooks below
        hooks.SaveCheckpointHook(interval=1, checkpoint_dir='./ckpt'),
        hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
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
