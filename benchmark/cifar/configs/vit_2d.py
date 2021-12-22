TOTAL_BATCH_SIZE = 512
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 3e-2

TENSOR_PARALLEL_SIZE = 4
TENSOR_PARALLEL_MODE = '2d'

NUM_EPOCHS = 200
WARMUP_EPOCHS = 40

parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

gradient_accumulation = 1

BATCH_SIZE = TOTAL_BATCH_SIZE // gradient_accumulation

gradient_clipping = 1.0

seed = 42

LOG_PATH = f"./vit_{TENSOR_PARALLEL_MODE}_cifar10_tp{TENSOR_PARALLEL_SIZE}_bs{TOTAL_BATCH_SIZE}_lr{LEARNING_RATE}_clip_grad{gradient_clipping}/"
