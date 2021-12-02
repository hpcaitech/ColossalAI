import glob
import os
import torch
import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_global_dist_logger
from colossalai.trainer import Trainer
from colossalai.utils import set_global_multitimer_status
from dataloader.dataloader import CIFAR10Dataset_SimCLR


def build_dali_train():
    train_dataset = CIFAR10Dataset_SimCLR(gpc.config.transform_cfg,
                                    root=gpc.config.dali.root, 
                                    train=True)
    return torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True, 
        num_workers = 8,
        batch_size=gpc.config.BATCH_SIZE,
    )

def build_dali_test():
    val_dataset = CIFAR10Dataset_SimCLR(gpc.config.transform_cfg,
                                    root=gpc.config.dali.root, 
                                    train=False)
    return torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False, 
        num_workers = 8,
        batch_size=gpc.config.BATCH_SIZE,
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
