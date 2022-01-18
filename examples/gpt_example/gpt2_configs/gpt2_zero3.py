from model.gpt import GPT2_small
from torch.optim import Adam
from deepspeed.ops.adam import DeepSpeedCPUAdam


BATCH_SIZE = 2
NUM_EPOCHS = 60
SEQ_LEN = 1024

loss_args = dict(
    init_scale=2**8,
)

zero = dict(
    level=3,
    dynamic_loss_scale=True,
    overlap_comm=True,
    clip_grad=1.0,
    offload_optimizer_config=dict(
        device='cpu',
        pin_memory=True,
        fast_init=False,
    ),
    offload_param_config=dict(
        device='cpu',
        pin_memory=True,
        max_in_cpu=1e9,
    )
)


optimizer = dict(
    type=DeepSpeedCPUAdam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=GPT2_small,
    checkpoint=True,
)
