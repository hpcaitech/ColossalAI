import glob
import os
import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.trainer import Trainer
from colossalai.utils import set_global_multitimer_status
from dataloader.imagenet_dali_dataloader import DaliDataloader


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
    engine, train_dataloader, test_dataloader = colossalai.initialize(
        train_dataloader=build_dali_train,
        test_dataloader=build_dali_test
    )
    logger = get_dist_logger()
    set_global_multitimer_status(True)
    timer = colossalai.utils.get_global_multitimer()
    trainer = Trainer(engine=engine,
                      verbose=True,
                      timer=timer)

    trainer.fit(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        hooks_cfg=gpc.config.hooks,
        display_progress=True,
        test_interval=1
    )


if __name__ == '__main__':
    main()
