from colossalai.amp import AMP_TYPE


LOG_NAME = 'cifar-simclr' 

BATCH_SIZE = 512
NUM_EPOCHS = 801
LEARNING_RATE = 0.03*BATCH_SIZE/256
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9


fp16 = dict(
    mode=AMP_TYPE.TORCH,
)

dataset = dict(
    root='./dataset',
)

gradient_accumulation=2
gradient_clipping=1.0

