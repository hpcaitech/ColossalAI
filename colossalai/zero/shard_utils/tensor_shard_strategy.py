from colossalai.zero.shard_utils import BaseShardStrategy
import torch.distributed as dist
from typing import List, Optional
from colossalai.zero.sharded_param.sharded_tensor import ShardedTensor


class TensorShardStrategy(BaseShardStrategy):

    def __init__(self, process_group: Optional[dist.ProcessGroup] = None) -> None:
        super().__init__(process_group)

    def shard(self, tensor_list: List[ShardedTensor]):
        for t in tensor_list:
            t.shard()

    def gather(self, tensor_list: List[ShardedTensor]):
        for t in tensor_list:
            t.gather()
