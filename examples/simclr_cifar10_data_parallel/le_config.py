from colossalai.amp import AMP_TYPE


LOG_NAME = 'cifar-simclr'
EPOCH = 800

BATCH_SIZE = 512
NUM_EPOCHS = 51
LEARNING_RATE = 0.03*BATCH_SIZE/256
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9


fp16 = dict(
    mode=AMP_TYPE.TORCH,
)

dataset = dict(
    root='./dataset',
)

gradient_accumulation=1
clip_grad_norm=1.0
