from colossalai.engine import AMP_TYPE
from torch.nn import CrossEntropyLoss
from mixup import MixupLoss
from hooks import TotalBatchsizeHook
from colossalai.registry import MODELS
from timm.models import vit_base_patch16_224

MODELS.register_module(vit_base_patch16_224)

LOG_NAME = 'vit-b16-1k-32k-mixup-light2'
# ViT Base
BATCH_SIZE = 256
DROP_RATE = 0.1
NUM_EPOCHS = 300

parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=1, mode=None),
)

optimizer = dict(
    type='Lamb',
    lr=1.8e-2,
    weight_decay=0.1,
)


loss = dict(
    type='MixupLoss',
    loss_fn_cls=CrossEntropyLoss
)

model = dict(
    type='vit_base_patch16_224',
    drop_rate=DROP_RATE,
)

hooks = [
    dict(type='LogMetricByEpochHook'),
    dict(type='AccuracyHook'),
    dict(type='LossHook'),
    dict(type='TotalBatchsizeHook'),
    dict(type='TensorboardHook', log_dir=f'./tb_logs/{LOG_NAME}'),
    dict(type='SaveCheckpointHook', interval=1,
         checkpoint_dir=f'./ckpt/{LOG_NAME}'),
    # dict(type='LoadCheckpointHook', epoch=10,
    #      checkpoint_dir=f'./ckpt/{LOG_NAME}'),
    dict(
        type='LRSchedulerHook',
        by_epoch=True,
        lr_scheduler_cfg=dict(
            type='LinearWarmupLR',
            warmup_steps=150
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
    root='./dataset/ILSVRC2012_1k',
    gpu_aug=True,
    mixup_alpha=0.2
)

engine = dict(
    schedule=None,
    gradient_handlers=None,
    gradient_accumulation=32,
    gradient_clipping=1.0,
)
