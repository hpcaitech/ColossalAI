from model import GPT3_pipeline_hybrid
from torch.optim import Adam
from colossalai.amp import AMP_TYPE
import torch


BATCH_SIZE = 2*48
NUM_EPOCHS = 60
SEQ_LEN = 2048
NUM_MICRO_BATCHES = 48
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES // 2, SEQ_LEN, 12288 // 2)

fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

parallel = dict(
    pipeline=24,
    tensor=dict(mode='2.5d', depth = 1, size=4)
)

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=GPT3_pipeline_hybrid,
    checkpoint=True,
    dtype=torch.half,
    num_chunks=1,
)
