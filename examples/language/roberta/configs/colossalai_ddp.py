from colossalai.nn.optimizer import FusedAdam

try:
    from colossalai.zero.shard_utils import TensorShardStrategy
except ImportError:
    # colossalai > 0.2.8
    from colossalai.zero.legacy import TensorShardStrategy

clip_grad_norm = 1.0
