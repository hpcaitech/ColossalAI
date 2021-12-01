from colossalai.engine import AMP_TYPE
from NT_Xentloss import NT_Xentloss
from hooks import TotalBatchsizeHook
from colossalai.registry import MODELS
from models.simclr import SimCLR


MODELS.register_module(SimCLR)

LOG_NAME = 'cifar-simclr'  # 'debug' #  

BATCH_SIZE = 512
NUM_EPOCHS = 801

parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=1, mode=None),
)

optimizer = dict(
    type='FusedSGD',
    lr=0.03*BATCH_SIZE/256,
    weight_decay=1.5e-6,
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
    root='../../../../../datasets/cifar10',
    train_path='cifar10_train.tfrecord',
    train_idx_path='cifar10_train.tfrecord.idx',
    val_path='cifar10_test.tfrecord',
    val_idx_path='cifar10_test.tfrecord.idx',
    gpu_aug=True,
    resize=32,
    crop=32,
    # mean_std=[[0.4914*255, 0.4822*255, 0.4465*255], [0.2023*255, 0.1994*255, 0.2010*255]]
    mean_std=[[127.5], [127.5]]
)

engine = dict(
    schedule=None,
    gradient_handlers=None,
    gradient_accumulation=1,
    gradient_clipping=1.0,
)
