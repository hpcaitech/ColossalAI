from colossalai.amp import AMP_TYPE

# hyperparameters
# BATCH_SIZE is as per GPU
# global batch size = BATCH_SIZE x data parallel size
BATCH_SIZE = 256
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.3
NUM_EPOCHS = 10
WARMUP_EPOCHS = 3

# model config
IMG_SIZE = 224
PATCH_SIZE = 16
HIDDEN_SIZE = 512
DEPTH = 4
NUM_HEADS = 4
MLP_RATIO = 2
NUM_CLASSES = 1000
CHECKPOINT = False
SEQ_LENGTH = (IMG_SIZE // PATCH_SIZE)**2 + 1    # add 1 for cls token

# parallel setting
TENSOR_PARALLEL_SIZE = 1
TENSOR_PARALLEL_MODE = '1d'

parallel = dict(
    tensor=dict(size=4, mode='sequence')
)

fp16 = dict(mode=AMP_TYPE.NAIVE)
clip_grad_norm = 1.0

# pipeline config
NUM_MICRO_BATCHES = parallel['pipeline']
