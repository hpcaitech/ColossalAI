from titans.loss.lm_loss import GPTLMLoss
from titans.model.gpt import gpt2_small
from torch.optim import Adam

from colossalai.amp import AMP_TYPE

BATCH_SIZE = 4
SEQ_LEN = 1024
NUM_EPOCHS = 60
TENSOR_PARALLEL = 4

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

fp16 = dict(mode=AMP_TYPE.NAIVE)

loss = dict(type=GPTLMLoss,)

model = dict(
    type=gpt2_small,
    checkpoint=True,
)

parallel = dict(
    pipeline=1,
    tensor=dict(size=TENSOR_PARALLEL, mode='2d'),
)
