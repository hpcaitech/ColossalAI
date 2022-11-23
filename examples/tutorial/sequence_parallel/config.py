from colossalai.amp import AMP_TYPE

DATA_PATH = ''
VOCAB_FILE_PATH = ''

# hyper-parameters
TRAIN_ITERS = 1000000
DECAY_ITERS = 990000
WARMUP_FRACTION = 0.01
GLOBAL_BATCH_SIZE = 32    # dp world size * sentences per GPU
EVAL_ITERS = 10
EVAL_INTERVAL = 10
LR = 0.0001
MIN_LR = 1e-05
WEIGHT_DECAY = 0.01
SEQ_LENGTH = 512

# BERT config
DEPTH = 12
NUM_ATTENTION_HEADS = 12
HIDDEN_SIZE = 768

# model config
ADD_BINARY_HEAD = False

# random seed
SEED = 1234

# pipeline config
# only enabled when pipeline > 1
NUM_MICRO_BATCHES = 4

# colossalai config
parallel = dict(pipeline=1, tensor=dict(size=2, mode='sequence'))

fp16 = dict(mode=AMP_TYPE.NAIVE, verbose=True)

gradient_handler = [dict(type='SequenceParallelGradientHandler')]
