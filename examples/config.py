from colossalai.amp import AMP_TYPE

BATCH_SIZE = 256
NUM_EPOCHS = 300

CONFIG = dict(
    fp16=dict(
        mode=AMP_TYPE.TORCH
    )
)