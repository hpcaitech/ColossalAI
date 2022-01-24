from colossalai.amp import AMP_TYPE

BATCH_SIZE = 64
NUM_EPOCHS = 100

CONFIG = dict(
    fp16=dict(
        mode=AMP_TYPE.TORCH
    )
)