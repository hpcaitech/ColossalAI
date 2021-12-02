from colossalai.engine import AMP_TYPE
from torch.nn import CrossEntropyLoss
from hooks import TotalBatchsizeHook
from torchvision.transforms import transforms
from colossalai.registry import MODELS
from models.linear_eval import Linear_eval


MODELS.register_module(Linear_eval)

LOG_NAME = 'cifar-simclr35'
EPOCH = 800

BATCH_SIZE = 512
NUM_EPOCHS = 50

parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=1, mode=None),
)

transform_cfg = [
    dict(type='RandomResizedCrop',
        size=32,
        scale=(0.2, 1.0)),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize',
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]),
]


optimizer = dict(
    type='FusedSGD',
    lr=0.03*BATCH_SIZE/256,
    weight_decay=0,
    momentum = 0.9
)

loss = dict(
    type='CrossEntropyLoss'
)

model = dict(
    type='Linear_eval',
    model='resnet18',
    class_num=10
)

hooks = [
    dict(type='LogMetricByEpochHook'),
    dict(type='AccuracyHook'),
    dict(type='LossHook'),
    dict(type='TotalBatchsizeHook'),
    dict(type='TensorboardHook', log_dir=f'./tb_logs/{LOG_NAME}-eval'),
    dict(type='SaveCheckpointHook', interval=20,
         checkpoint_dir=f'./ckpt/{LOG_NAME}-eval'),
    # dict(type='LoadCheckpointHook', epoch=750,
    #      checkpoint_dir=f'./ckpt/{LOG_NAME}'),
    dict(
        type='LRSchedulerHook',
        by_epoch=True,
        lr_scheduler_cfg=dict(
            type='CosineAnnealingWarmupLR',
            warmup_steps=5
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
    root='../../../../../datasets',
)

engine = dict(
    schedule=None,
    gradient_handlers=None,
    gradient_accumulation=1,
    gradient_clipping=1.0,
)
