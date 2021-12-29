#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import glob
import os

import colossalai
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec
import torch
from colossalai.builder import *
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import Accuracy, CrossEntropyLoss
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.trainer import Trainer
from colossalai.trainer.hooks import (AccuracyHook, LogMemoryByEpochHook, LogMetricByEpochHook, LogMetricByStepHook,
                                      LogTimingByEpochHook, LossHook, LRSchedulerHook, ThroughputHook)
from colossalai.utils import MultiTimer
from model_zoo.vit import vit_small_patch16_224
from nvidia.dali import types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

DATASET_PATH = str(os.environ['DATA'])

TRAIN_RECS = DATASET_PATH + '/train/*'
VAL_RECS = DATASET_PATH + '/validation/*'
TRAIN_IDX = DATASET_PATH + '/idx_files/train/*'
VAL_IDX = DATASET_PATH + '/idx_files/validation/*'


class DaliDataloader(DALIClassificationIterator):
    def __init__(self,
                 tfrec_filenames,
                 tfrec_idx_filenames,
                 shard_id=0,
                 num_shards=1,
                 batch_size=128,
                 num_threads=4,
                 resize=256,
                 crop=224,
                 prefetch=2,
                 training=True,
                 gpu_aug=False,
                 cuda=True):
        pipe = Pipeline(batch_size=batch_size,
                        num_threads=num_threads,
                        device_id=torch.cuda.current_device() if cuda else None,
                        seed=1024)
        with pipe:
            inputs = fn.readers.tfrecord(path=tfrec_filenames,
                                         index_path=tfrec_idx_filenames,
                                         random_shuffle=training,
                                         shard_id=shard_id,
                                         num_shards=num_shards,
                                         initial_fill=10000,
                                         read_ahead=True,
                                         prefetch_queue_depth=prefetch,
                                         name='Reader',
                                         features={
                                             'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ""),
                                             'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                                         })
            images = inputs["image/encoded"]

            if training:
                images = fn.decoders.image(images, device='mixed' if gpu_aug else 'cpu', output_type=types.RGB)
                images = fn.random_resized_crop(images, size=crop, device='gpu' if gpu_aug else 'cpu')
                flip_lr = fn.random.coin_flip(probability=0.5)
            else:
                # decode jpeg and resize
                images = fn.decoders.image(images, device='mixed' if gpu_aug else 'cpu', output_type=types.RGB)
                images = fn.resize(images,
                                   device='gpu' if gpu_aug else 'cpu',
                                   resize_x=resize,
                                   resize_y=resize,
                                   dtype=types.FLOAT,
                                   interp_type=types.INTERP_TRIANGULAR)
                flip_lr = False

            # center crop and normalise
            images = fn.crop_mirror_normalize(images,
                                              dtype=types.FLOAT,
                                              crop=(crop, crop),
                                              mean=[127.5],
                                              std=[127.5],
                                              mirror=flip_lr)
            label = inputs["image/class/label"] - 1  # 0-999
            # LSG: element_extract will raise exception, let's flatten outside
            # label = fn.element_extract(label, element_map=0)  # Flatten
            if cuda:  # transfer data to gpu
                pipe.set_outputs(images.gpu(), label.gpu())
            else:
                pipe.set_outputs(images, label)

        pipe.build()
        last_batch_policy = 'DROP' if training else 'PARTIAL'
        super().__init__(pipe, reader_name="Reader", auto_reset=True, last_batch_policy=last_batch_policy)

    def __iter__(self):
        # if not reset (after an epoch), reset; if just initialize, ignore
        if self._counter >= self._size or self._size < 0:
            self.reset()
        return self

    def __next__(self):
        data = super().__next__()
        img, label = data[0]['data'], data[0]['label']
        label = label.squeeze()
        return (img, ), (label, )


def build_dali_train(batch_size):
    return DaliDataloader(
        sorted(glob.glob(TRAIN_RECS)),
        sorted(glob.glob(TRAIN_IDX)),
        batch_size=batch_size,
        shard_id=gpc.get_local_rank(ParallelMode.DATA),
        num_shards=gpc.get_world_size(ParallelMode.DATA),
        training=True,
        gpu_aug=True,
        cuda=True,
    )


def build_dali_test(batch_size):
    return DaliDataloader(
        sorted(glob.glob(VAL_RECS)),
        sorted(glob.glob(VAL_IDX)),
        batch_size=batch_size,
        shard_id=gpc.get_local_rank(ParallelMode.DATA),
        num_shards=gpc.get_world_size(ParallelMode.DATA),
        training=False,
        gpu_aug=True,
        cuda=True,
    )


def train_imagenet():
    args = colossalai.get_default_parser().parse_args()
    # standard launch
    # colossalai.launch(config=args.config,
    #                   rank=args.rank,
    #                   world_size=args.world_size,
    #                   local_rank=args.local_rank,
    #                   host=args.host,
    #                   port=args.port)

    # launch from torchrun
    colossalai.launch_from_torch(config=args.config)

    logger = get_dist_logger()
    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    model = vit_small_patch16_224(num_classes=100, init_method='jax')

    train_dataloader = build_dali_train(gpc.config.BATCH_SIZE // gpc.data_parallel_size)
    test_dataloader = build_dali_test(gpc.config.BATCH_SIZE // gpc.data_parallel_size)

    criterion = CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)

    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,
                                           total_steps=gpc.config.NUM_EPOCHS,
                                           warmup_steps=gpc.config.WARMUP_EPOCHS)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model=model,
                                                                         optimizer=optimizer,
                                                                         criterion=criterion,
                                                                         train_dataloader=train_dataloader,
                                                                         test_dataloader=test_dataloader)

    logger.info("Engine is built", ranks=[0])

    timer = MultiTimer()

    trainer = Trainer(engine=engine, logger=logger, timer=timer)
    logger.info("Trainer is built", ranks=[0])

    hooks = [
        LogMetricByEpochHook(logger=logger),
        LogMetricByStepHook(),
        # LogTimingByEpochHook(timer=timer, logger=logger),
        # LogMemoryByEpochHook(logger=logger),
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
    train_imagenet()
