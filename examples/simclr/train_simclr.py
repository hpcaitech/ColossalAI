import glob
import os
import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_global_dist_logger
from colossalai.trainer import Trainer
from colossalai.utils import set_global_multitimer_status
from dataloader.dali_dataloader import DALIDataLoader


def build_dali_train():
    root = gpc.config.dali.root
    train_pat = os.path.join(root, gpc.config.dali.train_path)
    train_idx_pat = os.path.join(root, gpc.config.dali.train_idx_path)
    return DALIDataLoader(
        sorted(glob.glob(train_pat)),
        sorted(glob.glob(train_idx_pat)),
        batch_size=gpc.config.BATCH_SIZE,
        shard_id=gpc.get_local_rank(ParallelMode.DATA),
        num_shards=gpc.get_world_size(ParallelMode.DATA),
        training=True,
        gpu_aug=gpc.config.dali.gpu_aug,
        cuda=True,
        resize=gpc.config.dali.resize,
        crop=gpc.config.dali.crop,
        mean_std=gpc.config.dali.mean_std
    )


def build_dali_test():
    root = gpc.config.dali.root
    val_pat = os.path.join(root, gpc.config.dali.val_path)
    val_idx_pat = os.path.join(root, gpc.config.dali.val_idx_path)
    return DALIDataLoader(
        sorted(glob.glob(val_pat)),
        sorted(glob.glob(val_idx_pat)),
        batch_size=gpc.config.BATCH_SIZE,
        shard_id=gpc.get_local_rank(ParallelMode.DATA),
        num_shards=gpc.get_world_size(ParallelMode.DATA),
        training=False,
        # gpu_aug=gpc.config.dali.gpu_aug,
        gpu_aug=False,
        cuda=True,
        resize=gpc.config.dali.resize,
        crop=gpc.config.dali.crop,
        mean_std=gpc.config.dali.mean_std
    )


def main():
    engine, train_dataloader, test_dataloader = colossalai.initialize(
        train_dataloader=build_dali_train,
        test_dataloader=build_dali_test
    )
    logger = get_global_dist_logger()
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
