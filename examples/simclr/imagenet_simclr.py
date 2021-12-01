from colossalai.engine import AMP_TYPE
from NT_Xentloss import NT_Xentloss
from hooks import TotalBatchsizeHook
from colossalai.registry import MODELS
from models.simclr import SimCLR


MODELS.register_module(SimCLR)

LOG_NAME = 'imagenet-simclr'

BATCH_SIZE = 128
NUM_EPOCHS = 300

parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=1, mode=None),
)

optimizer = dict(
    type='Lars',
    lr=0.03*BATCH_SIZE/256,
    weight_decay=1.5e-6,
    momentum = 0.9
)

loss = dict(
    type='NT_Xentloss'
)

model = dict(
    type='SimCLR',
    model='resnet50',
)

hooks = [
    dict(type='LogMetricByEpochHook'),
    dict(type='LossHook'),
    dict(type='TotalBatchsizeHook'),
    dict(type='TensorboardHook', log_dir=f'./tb_logs/{LOG_NAME}'),
    dict(type='SaveCheckpointHook', interval=10,
         checkpoint_dir=f'./ckpt/{LOG_NAME}'),
    dict(
        type='LRSchedulerHook',
        by_epoch=True,
        lr_scheduler_cfg=dict(
            type='CosineAnnealingWarmupLR',
            warmup_steps=10
        )
    ),
]

fp16 = dict(
    mode=AMP_TYPE.TORCH,
)


logging = dict(
    root_path=f"./logs/{LOG_NAME}"
)

dali = dict(
    root='../../../../../datasets/imagenet-scratch',
    train_path='train/*',
    train_idx_path='train_idx_files/*',
    val_path='validation/*',
    val_idx_path='validation_idx_files/*',
    gpu_aug=True,
    resize=224,
    crop=224,
    # mean_std=[[0.485*255, 0.456*255, 0.406*255], [0.229*255, 0.224*255, 0.225*255]]
    mean_std=[[127.5], [127.5]]
)

engine = dict(
    schedule=None,
    gradient_handlers=None,
    gradient_accumulation=16,
    gradient_clipping=1.0,
)
