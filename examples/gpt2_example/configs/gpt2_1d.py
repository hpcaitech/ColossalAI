from colossalai.amp import AMP_TYPE
from model.gpt import GPTLMLoss, GPT2_small
from torch.optim import Adam


BATCH_SIZE = 1
SEQ_LEN = 1024
NUM_EPOCHS = 60

TENSOR_PARALLEL = 2

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

loss = dict(
    type=GPTLMLoss,
)

model = dict(
    type=GPT2_small,
    checkpoint=True,
)

hooks = [
    dict(type='LogMetricByEpochHook'),
    dict(type='LossHook'),
    dict(
        type='LRSchedulerHook',
        by_epoch=True,
        lr_scheduler_cfg=dict(
            type='CosineAnnealingWarmupLR',
            warmup_steps=5
        )
    ),
]

parallel = dict(
    pipeline=1,
    tensor=dict(size=TENSOR_PARALLEL, mode='1d'),
)
