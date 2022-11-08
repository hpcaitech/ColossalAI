from model import GPT2_small_pipeline_hybrid

from colossalai.nn.optimizer import HybridAdam
from colossalai.zero.shard_utils import BucketTensorShardStrategy, TensorShardStrategy

BATCH_SIZE = 8
NUM_EPOCHS = 60
SEQ_LEN = 1024
NUM_MICRO_BATCHES = 4
HIDDEN_SIZE = 768
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LEN, HIDDEN_SIZE)
zero = dict(model_config=dict(tensor_placement_policy='cpu', shard_strategy=BucketTensorShardStrategy()),
            optimizer_config=dict())

optimizer = dict(
    type=HybridAdam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(type=GPT2_small_pipeline_hybrid, checkpoint=True, num_chunks=1)

parallel = dict(
    pipeline=2,
    tensor=dict(size=2, mode='1d'),
)
