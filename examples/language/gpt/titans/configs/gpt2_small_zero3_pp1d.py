from model import GPT2_small_pipeline_hybrid

from colossalai.nn.optimizer import HybridAdam
from colossalai.zero.shard_utils import TensorShardStrategy

BATCH_SIZE = 8
NUM_EPOCHS = 10
SEQ_LEN = 1024
NUM_MICRO_BATCHES = 4
HIDDEN_SIZE = 768
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LEN, HIDDEN_SIZE)

# if you do no want zero, just comment out this dictionary
zero = dict(
    model_config=dict(tensor_placement_policy="cuda", shard_strategy=TensorShardStrategy()),
    optimizer_config=dict(initial_scale=2**5),
)

optimizer = dict(
    type=HybridAdam,
    lr=0.000015,
    weight_decay=1e-2,
)

model = dict(type=GPT2_small_pipeline_hybrid, checkpoint=True, num_chunks=1)

# pipeline parallel: modify integer value for the number of pipeline stages
# tensor parallel: modify size to set the tensor parallel size, usually the number of GPUs per node
# for the current model implementation, mode can only be 1D or None
parallel = dict(
    pipeline=1,
    tensor=dict(size=2, mode="1d"),
)
