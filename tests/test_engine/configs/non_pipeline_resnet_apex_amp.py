import os
from pathlib import Path


BATCH_SIZE = 128
IMG_SIZE = 224
DIM = 768
NUM_CLASSES = 10
NUM_ATTN_HEADS = 12


parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=1, mode=None)
)
fp16 = dict(mode=AMP_TYPE.APEX)
