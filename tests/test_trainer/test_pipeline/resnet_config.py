import os
import model
from pathlib import Path

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
    pipeline=dict(size=4),
    tensor=dict(size=1, mode=None)
)
