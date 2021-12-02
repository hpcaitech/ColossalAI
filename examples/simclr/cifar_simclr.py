import types
from colossalai.engine import AMP_TYPE
from numpy.core.fromnumeric import size
from NT_Xentloss import NT_Xentloss
from hooks import TotalBatchsizeHook
from torchvision.transforms import transforms
from colossalai.registry import MODELS
from models.simclr import SimCLR


MODELS.register_module(SimCLR)

LOG_NAME = 'cifar-simclr35'  # 'debug' #  

BATCH_SIZE = 512
NUM_EPOCHS = 801

parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=1, mode=None),
)

transform_cfg = [
    dict(type='RandomResizedCrop',
        size=32,
        scale=(0.2, 1.0)),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomApply',
        transforms=[
        transforms.ColorJitter(0.8,0.8,0.8,0.2)
        # dict(type='ColorJitter', 
        # brightness=0.8,
        # contrast=0.8,
        # saturation=0.8,
        # hue=0.2)
        ],
        p=0.8),
    dict(type='RandomGrayscale',
        p=0.2),
    dict(type='RandomApply',
        transforms=[
        transforms.GaussianBlur(kernel_size=32//20*2+1, sigma=(0.1, 2.0))
        # dict(type='GaussianBlur',
        # kernel_size=32//20*2+1,
        # sigma=(0.1, 2.0))
        ],
        p=0.5),
    dict(type='ToTensor'),
    dict(type='Normalize',
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]),
]


optimizer = dict(
    type='FusedSGD',
    lr=0.03*BATCH_SIZE/256,
    weight_decay=0.0005,
    momentum = 0.9
)

loss = dict(
    type='NT_Xentloss'
)

model = dict(
    type='SimCLR',
    model='resnet18'
)

hooks = [
    dict(type='LogMetricByEpochHook'),
    dict(type='LossHook'),
    dict(type='TotalBatchsizeHook'),
    dict(type='TensorboardHook', log_dir=f'./tb_logs/{LOG_NAME}'),
    dict(type='SaveCheckpointHook', interval=50,
         checkpoint_dir=f'./ckpt/{LOG_NAME}'),
    # dict(type='LoadCheckpointHook', epoch=800,
    #      checkpoint_dir=f'./ckpt/{LOG_NAME}'),
    dict(
        type='LRSchedulerHook',
        by_epoch=True,
        lr_scheduler_cfg=dict(
            type='CosineAnnealingWarmupLR',
            warmup_steps=50
        )
    ),
]

# fp16 = dict(
#     mode=AMP_TYPE.TORCH,
# )

logging = dict(
    root_path=f"./logs/{LOG_NAME}"
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
