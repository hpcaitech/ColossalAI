from colossalai.legacy.amp import AMP_TYPE

# hyperparameters
# BATCH_SIZE is as per GPU
# global batch size = BATCH_SIZE x data parallel size
BATCH_SIZE = 512
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.3
NUM_EPOCHS = 2
WARMUP_EPOCHS = 1

# model config
NUM_CLASSES = 10

fp16 = dict(mode=AMP_TYPE.NAIVE)
clip_grad_norm = 1.0
