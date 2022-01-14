from colossalai.amp import AMP_TYPE
from model import GPTLMLoss, GPT2_small_pipeline_hybrid
from torch.optim import Adam


BATCH_SIZE = 1
SEQ_LEN = 1024
NUM_EPOCHS = 60
NUM_MICRO_BATCHES = 1
PIPELINE = 2

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
    type=GPT2_small_pipeline_hybrid,
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
    pipeline=PIPELINE,
    tensor=dict(size=1, mode=None),
)
