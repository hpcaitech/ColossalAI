from model import GPT2_exlarge_pipeline_1D
from torch.optim import Adam
from colossalai.amp import AMP_TYPE
import torch


BATCH_SIZE = 192
NUM_EPOCHS = 60
SEQ_LEN = 1024
NUM_MICRO_BATCHES = 192
HIDDEN_SIZE = 1600
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LEN, HIDDEN_SIZE)

fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

parallel = dict(
    pipeline=4,
    tensor=dict(mode='1d', size=1)
)

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=GPT2_exlarge_pipeline_1D,
    checkpoint=True,
    dtype=torch.half,
    # num_chunks=2,
)
