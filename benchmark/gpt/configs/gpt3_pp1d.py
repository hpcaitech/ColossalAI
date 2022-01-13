import torch
from colossalai.amp import AMP_TYPE
from colossalai.engine.schedule import InterleavedPipelineSchedule
from model_zoo.gpt import gpt3_pipeline

VOCAB_SIZE = 50304
SEQ_LENGTH = 2048

TOTAL_BATCH_SIZE = 192
LEARNING_RATE = 0.00015
WEIGHT_DECAY = 1e-2

gradient_accumulation = 1

clip_grad_norm = 1.0

BATCH_SIZE = TOTAL_BATCH_SIZE // gradient_accumulation

TENSOR_PARALLEL_SIZE = 4
TENSOR_PARALLEL_MODE = '1d'

PIPELINE_SIZE = 32
MICRO_BATCH_SIZE = 1
NUM_MICRO_BATCHES = BATCH_SIZE // MICRO_BATCH_SIZE
NUM_CHUNKS = 1

NUM_EPOCHS = 20
WARMUP_EPOCHS = int(NUM_EPOCHS * 0.36)

parallel = dict(
    pipeline=PIPELINE_SIZE,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

model = dict(type=gpt3_pipeline,
             num_chunks=NUM_CHUNKS,
             vocab_size=VOCAB_SIZE,
             max_position_embeddings=SEQ_LENGTH,
             dtype=torch.half,
             fuse_scale_mask_softmax=True,
             checkpoint=True)

schedule = dict(type=InterleavedPipelineSchedule,
                num_microbatches=NUM_MICRO_BATCHES,
                num_model_chunks=NUM_CHUNKS,
                tensor_shape=(MICRO_BATCH_SIZE, SEQ_LENGTH, 12288),
                scatter_gather_tensors=True)

fp16 = dict(mode=AMP_TYPE.NAIVE, )

# LOG_PATH = f"./gpt3_{TENSOR_PARALLEL_MODE}_pp{PIPELINE_SIZE}_tp{TENSOR_PARALLEL_SIZE}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_accum{gradient_accumulation}_clip_grad{clip_grad_norm}/"
