from colossalai.engine import AMP_TYPE
from model.gpt import GPTLMLoss, GPT2_exlarge
from dataset.webtext import WebtextDataset


BATCH_SIZE = 2
NUM_EPOCHS = 60

train_data = dict(
    dataset=dict(
        type='WebtextDataset',
        path='/project/scratch/p200012/dataset/openwebtext/small-gpt-dataset.json',
    ),
    dataloader=dict(
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
)

#zero = dict(
#    type='ZeroRedundancyOptimizer_Level_2',
#    dynamic_loss_scale=True,
#    overlap_comm=True,
#    clip_grad=1.0,
#    cpu_offload=True,
#)


optimizer = dict(
    type='Adam',
    lr=0.00015,
    weight_decay=1e-2,
)

loss = dict(
    type='GPTLMLoss',
)

model = dict(
    type='GPT2_exlarge',
    checkpoint=True,
)

hooks = [
    dict(type='LogMetricByEpochHook'),
    # dict(type='LogTimingByEpochHook'),
    # dict(type='LogMemoryByEpochHook'),
    dict(type='LossHook'),
    # dict(type='TensorboardHook', log_dir='./tfb_logs'),
    dict(
        type='LRSchedulerHook',
        by_epoch=True,
        lr_scheduler_cfg=dict(
            type='CosineAnnealingWarmupLR',
            warmup_steps=5
        )
    ),
    # dict(type='SaveCheckpointHook', interval=5, checkpoint_dir='./ckpt'),
    # dict(type='LoadCheckpointHook', epoch=20, checkpoint_dir='./ckpt')
]

parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=1, mode=None),
)

# logging = dict(
#     root_path='./logs'
# )
