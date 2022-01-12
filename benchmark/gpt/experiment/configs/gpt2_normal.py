
from colossalai.amp import AMP_TYPE
from model.gpt import GPTLMLoss, GPT2_small
from torch.optim import Adam


BATCH_SIZE = 2
NUM_EPOCHS = 60
SEQ_LEN = 1024
NUM_MICRO_BATCHES = 1

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
    type=GPT2_small,
    checkpoint=True,
)

parallel = dict(
    pipeline=1,
    tensor=dict(mode=None, size=1),
)