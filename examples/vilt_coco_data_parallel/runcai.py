import os
import copy
import pytorch_lightning as pl

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
import argparse
from colossalai.trainer import Trainer, hooks
from colossalai.nn.lr_scheduler import LinearWarmupLR

from vilt.config import ex
from vilt.modules import ViLTransformerSS_CAI
from vilt.datamodules.multitask_datamodule import MTDataModule
from vilt_schedule import viltSchedule


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    # pl.seed_everything(_config["seed"])
    config = '/work/zhangyq/ColossalAI/examples/vilt/configs/baseline.py'
    gpc.load_config(config)
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        colossalai.launch_from_openmpi(config=config,
        host='localhost',
        port='11455',
        backend='nccl')
    elif 'SLURM_PROCID' in os.environ:
        colossalai.launch_from_slurm(config=config,
        host='localhost',
        port='11455',
        backend='nccl')
    elif 'WORLD_SIZE' in os.environ:
        colossalai.launch_from_torch(config=config,
        host='localhost',
        port='11455',
        backend='nccl')
    else:
        colossalai.launch(
            config=config,
            host='localhost',
            port='11455',
            rank=0,
            world_size=4,
            backend='nccl')
    logger = get_dist_logger('root')
    logger.info('launched')
    dm = MTDataModule(_config, dist=False)
    dm.prepare_data()
    dm.setup(0)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    # val_loader = train_loader
    # test_loader = dm.test_dataloader()
    model = ViLTransformerSS_CAI(_config)

    def itm_mlm_loss(output):
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss
    #Initialize engine and trainer

    optim = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.99)
    )

    schedule=viltSchedule()
    def batch_data_process_func(sample):
        return sample,sample
    schedule.batch_data_process_func = batch_data_process_func
    lr_scheduler = LinearWarmupLR(optim, warmup_steps=50, total_steps=gpc.config.NUM_EPOCHS)
    criterion=itm_mlm_loss

    engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(model=model,
    optimizer=optim,
    criterion=criterion,
    train_dataloader=train_loader,
    test_dataloader=val_loader,
    verbose=True,)

    logger.info("engine is built", ranks=[0])

    trainer = Trainer(engine=engine,
            schedule=schedule, logger=logger)
    
    # hook_list = [
    #     hooks.LossHook(),
    #     hooks.AccuracyHook(),
    #     hooks.LogMetricByEpochHook(logger),
    #     hooks.LRSchedulerHook(lr_scheduler, by_epoch=True),

    #     # comment if you do not need to use the hooks below
    #     hooks.SaveCheckpointHook(interval=1, checkpoint_dir='./ckpt'),
    #     hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
    # ]

    logger.info("trainer is built", ranks=[0])



    colossalai.context.config.Config.from_file(config)

    logger.info("start training", ranks=[0])
    trainer.fit(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=gpc.config.num_epochs,
        display_progress=True,
        test_interval=2
    )

    # exp_name = f'{_config["exp_name"]}'

    # os.makedirs(_config["log_dir"], exist_ok=True)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     save_top_k=1,
    #     verbose=True,
    #     monitor="val/the_metric",
    #     mode="max",
    #     save_last=True,
    # )
    # logger = pl.loggers.TensorBoardLogger(
    #     _config["log_dir"],
    #     name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    # )

    # lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    # callbacks = [checkpoint_callback, lr_callback]

    # num_gpus = (
    #     _config["num_gpus"]
    #     if isinstance(_config["num_gpus"], int)
    #     else len(_config["num_gpus"])
    # )

    # grad_steps = _config["batch_size"] // (
    #     _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    # )

    # max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    # trainer = pl.Trainer(
    #     gpus=_config["num_gpus"],
    #     num_nodes=_config["num_nodes"],
    #     precision=_config["precision"],
    #     accelerator="ddp",
    #     benchmark=True,
    #     deterministic=True,
    #     max_epochs=_config["max_epoch"] if max_steps is None else 1000,
    #     max_steps=max_steps,
    #     callbacks=callbacks,
    #     logger=logger,
    #     prepare_data_per_node=False,
    #     replace_sampler_ddp=False,
    #     accumulate_grad_batches=grad_steps,
    #     log_every_n_steps=10,
    #     flush_logs_every_n_steps=10,
    #     resume_from_checkpoint=_config["resume_from"],
    #     weights_summary="top",
    #     fast_dev_run=_config["fast_dev_run"],
    #     val_check_interval=_config["val_check_interval"],
    # )

    # if not _config["test_only"]:
    #     trainer.fit(model, datamodule=dm)
    # else:
    #     trainer.test(model, datamodule=dm)

# if __name__ == '__main__':
#     main()