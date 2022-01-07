from model.gpt import GPT2_exlarge
from torch.optim import Adam
from deepspeed.ops.adam import DeepSpeedCPUAdam


BATCH_SIZE = 2
NUM_EPOCHS = 60
EQ_LEN = 1024

zero = dict(
    level=3,
    dynamic_loss_scale=True,
    overlap_comm=True,
    clip_grad=1.0,
    offload_optimizer_config=dict(
        pin_memory=False,
        fast_init=False,
        device='nvme',
        nvme_path='/mnt/tier0',
        pipeline=False,
        buffer_count=4,
    ),
    offload_param_config=dict(
        pin_memory=False,
        max_in_cpu=1e9,
        device='nvme',
        nvme_path='/mnt/tier0',
        buffer_count=5,
        buffer_size=1e8,
    ),
    aio_config=dict(
        block_size=1048576,
        queue_depth=8,
        thread_count=1,
        single_submit=False,
        overlap_events=True
    )
)


optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=GPT2_exlarge,
    checkpoint=True,
)
