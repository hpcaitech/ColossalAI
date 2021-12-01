from colossalai.engine import AMP_TYPE
from torch.nn import CrossEntropyLoss
from hooks import TotalBatchsizeHook
from colossalai.registry import MODELS
from models.linear_eval import Linear_eval


MODELS.register_module(Linear_eval)

LOG_NAME = 'imagenet-simclr' # same as the log name of the simclr self-supervised training

PT_EPOCH = 800 # specify which epoch of the pretrained model to load


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
    type='CrossEntropyLoss'
)

model = dict(
    type='Linear_eval',
    model='resnet50',
    class_num=1000
)

hooks = [
    dict(type='LogMetricByEpochHook'),
    dict(type='AccuracyHook'),
    dict(type='LossHook'),
    dict(type='TotalBatchsizeHook'),
    dict(type='TensorboardHook', log_dir=f'./tb_logs/{LOG_NAME}-eval'),
    dict(type='SaveCheckpointHook', interval=10,
         checkpoint_dir=f'./ckpt/{LOG_NAME}-eval'),
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
    root_path=f"./logs/{LOG_NAME}-eval"
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
