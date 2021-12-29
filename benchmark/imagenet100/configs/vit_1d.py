from colossalai.amp import AMP_TYPE

TOTAL_BATCH_SIZE = 4096
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.3

TENSOR_PARALLEL_SIZE = 4
TENSOR_PARALLEL_MODE = '1d'

NUM_EPOCHS = 300
WARMUP_EPOCHS = 32

parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

fp16 = dict(mode=AMP_TYPE.TORCH, )

gradient_accumulation = 2

BATCH_SIZE = TOTAL_BATCH_SIZE // gradient_accumulation

clip_grad_norm = 1.0

LOG_PATH = f"./vit_{TENSOR_PARALLEL_MODE}_imagenet100_tp{TENSOR_PARALLEL_SIZE}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{fp16['mode']}_clip_grad{clip_grad_norm}/"
