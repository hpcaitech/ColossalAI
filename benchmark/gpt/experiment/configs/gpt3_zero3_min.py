from model.gpt import GPT3
from torch.optim import Adam
from deepspeed.ops.adam import DeepSpeedCPUAdam


BATCH_SIZE = 3
NUM_EPOCHS = 60
SEQ_LEN = 2048

zero = dict(
    level=3,
    dynamic_loss_scale=True,
    overlap_comm=True,
    clip_grad=1.0,
    offload_optimizer_config=dict(
        device='cpu',
        pin_memory=True,
        fast_init=True,
    ),
    offload_param_config=dict(
        device='cpu',
        pin_memory=True,
        max_in_cpu=1e9,
    ),
    # reduce_bucket_size=12288*12288,
    # prefetch_bucket_size=0.9*12288*12288,
    # param_persistence_threshold=10*12288,
    # sub_group_size=2e12,
)


optimizer = dict(
    type=DeepSpeedCPUAdam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=GPT3,
    checkpoint=True,
)
