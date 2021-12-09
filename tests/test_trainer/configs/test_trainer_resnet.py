import os
from pathlib import Path


hooks = [
    dict(type='LogMetricByEpochHook'),
    dict(type='AccuracyHook'),
    dict(type='LossHook'),
    dict(type='TensorboardHook', log_dir='./tfb_logs'),
    dict(
        type='LRSchedulerHook',
        by_epoch=True,
        lr_scheduler_cfg=dict(
            type='CosineAnnealingLR',
            warmup_steps=5
        )
    ),
    dict(type='SaveCheckpointHook', interval=5, checkpoint_dir='./ckpt'),
]
