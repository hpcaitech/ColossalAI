IMG_SIZE = 32
PATCH_SIZE = 4
HIDDEN_SIZE = 256
MLP_RATIO = 2
NUM_HEADS = 4
NUM_CLASSES = 10
DROP_RATE = 0.1
DEPTH = 7

BATCH_SIZE = 512
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 3e-2

TENSOR_PARALLEL_SIZE = 1
TENSOR_PARALLEL_MODE = None

parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

from colossalai.amp import AMP_TYPE
fp16 = dict(mode=AMP_TYPE.TORCH, )

gradient_accumulation = 1

gradient_clipping = 1.0

num_epochs = 200

warmup_epochs = 40

log_path = f"./vit_{TENSOR_PARALLEL_MODE}_cifar10_tp{TENSOR_PARALLEL_SIZE}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_clip_grad{gradient_clipping}/"

seed = 42
