import torch
from colossalai.amp import AMP_TYPE
from model_zoo.gpt import gpt2_8B, gpt2_xl, gpt2_medium, gpt2_large

VOCAB_SIZE = 50304
SEQ_LENGTH = 1024

TOTAL_BATCH_SIZE = 32
LEARNING_RATE = 0.00015
WEIGHT_DECAY = 1e-2

TENSOR_PARALLEL_SIZE = 4
TENSOR_PARALLEL_MODE = '2d'

NUM_EPOCHS = 20
WARMUP_EPOCHS = 1

parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

model = dict(type=gpt2_xl,
             vocab_size=VOCAB_SIZE,
             max_position_embeddings=SEQ_LENGTH,
             dtype=torch.half,
            #  fuse_scale_mask_softmax=True,
             checkpoint=True)

fp16 = dict(mode=AMP_TYPE.NAIVE)

gradient_accumulation = 1

BATCH_SIZE = TOTAL_BATCH_SIZE // gradient_accumulation

clip_grad_norm = 1.0

# LOG_PATH = f"./gpt2_{TENSOR_PARALLEL_MODE}_tp{TENSOR_PARALLEL_SIZE}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_accum{gradient_accumulation}_clip_grad{clip_grad_norm}/"
