from colossalai.amp import AMP_TYPE

VOCAB_SIZE = 50304
SEQ_LENGTH = 1024

TOTAL_BATCH_SIZE = 64
LEARNING_RATE = 0.00015
WEIGHT_DECAY = 1e-2

TENSOR_PARALLEL_SIZE = 1
TENSOR_PARALLEL_MODE = None

NUM_EPOCHS = 60
WARMUP_EPOCHS = int(NUM_EPOCHS * 0.36)

parallel = dict(
    pipeline=4,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

fp16 = dict(mode=AMP_TYPE.NAIVE, )

gradient_accumulation = 1

BATCH_SIZE = TOTAL_BATCH_SIZE // gradient_accumulation

NUM_MICRO_BATCHES = 16
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LENGTH, 1600)

clip_grad_norm = 1.0

# LOG_PATH = f"./gpt3_{TENSOR_PARALLEL_MODE}_tp{TENSOR_PARALLEL_SIZE}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_accum{gradient_accumulation}_clip_grad{clip_grad_norm}/"
