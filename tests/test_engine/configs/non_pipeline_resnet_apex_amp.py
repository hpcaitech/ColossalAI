import os
from pathlib import Path

from colossalai.engine import AMP_TYPE

BATCH_SIZE = 128
IMG_SIZE = 224
DIM = 768
NUM_CLASSES = 10
NUM_ATTN_HEADS = 12

# resnet 18
model = dict(type='VanillaResNet',
             block_type='ResNetBasicBlock',
             layers=[2, 2, 2, 2],
             num_cls=10)

parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=1, mode=None)
)

train_data = dict(dataset=dict(type='CIFAR10Dataset',
                               root=Path(os.environ['DATA']),
                               download=True,
                               transform_pipeline=[
                                   dict(type='Resize',
                                        size=(IMG_SIZE, IMG_SIZE)),
                                   dict(type='ToTensor'),
                                   dict(type='Normalize',
                                        mean=(0.5, 0.5, 0.5),
                                        std=(0.5, 0.5, 0.5))
                               ]),
                  dataloader=dict(batch_size=BATCH_SIZE,
                                  pin_memory=True,
                                  num_workers=4,
                                  drop_last=True))

optimizer = dict(type='Adam', lr=0.001)

loss = dict(type='CrossEntropyLoss')
fp16 = dict(mode=AMP_TYPE.APEX)

# set_device_func = lambda global_rank, world_size: global_rank % 4
seed = 1024
