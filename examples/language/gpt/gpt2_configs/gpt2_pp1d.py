import torch
from titans.loss.lm_loss import GPTLMLoss
from titans.loss.vocab_cross_entropy import vocab_parallel_cross_entropy
from titans.model.gpt import gpt2_small
from torch.optim import Adam

from colossalai.amp import AMP_TYPE

BATCH_SIZE = 8
NUM_EPOCHS = 60
SEQ_LEN = 1024

NUM_MICRO_BATCHES = 4
HIDDEN_SIZE = 768
PIPELINE = 2
TENSOR_PARALLEL = 2
MODE = '1d'

fp16 = dict(mode=AMP_TYPE.NAIVE)

parallel = dict(pipeline=PIPELINE, tensor=dict(mode=MODE, size=TENSOR_PARALLEL))

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=gpt2_small,
    checkpoint=True,
    dtype=torch.half,
)

loss_fn = dict(type=vocab_parallel_cross_entropy)
