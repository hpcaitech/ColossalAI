import torch
from model import GPT3_pipeline_1D
from torch.optim import Adam
from colossalai.amp import AMP_TYPE


BATCH_SIZE = 192
NUM_EPOCHS = 60
SEQ_LEN = 2048
NUM_MICRO_BATCHES = 192
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LEN, 12288)
#LOG_PATH = ''

fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

parallel = dict(
    pipeline=32,
    tensor=dict(mode='1d', size=4)
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
    # act_func='fused_gelu',
    # num_chunks=3,
)
