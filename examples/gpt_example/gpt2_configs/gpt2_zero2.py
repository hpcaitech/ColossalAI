from model.gpt import GPT2_small
from torch.optim import Adam


BATCH_SIZE = 1
NUM_EPOCHS = 60
SEQ_LEN = 1024

zero = dict(
    level=2,
    dynamic_loss_scale=True,
    overlap_comm=True,
    clip_grad=1.0,
    cpu_offload=True,
)


optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=GPT2_small,
    checkpoint=True,
)
