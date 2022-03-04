from tokenize import group
import torch
from colossalai.zero.shard_utils import BaseShardStrategy
import torch.distributed as dist
from typing import List, Optional
from torch import Tensor
from colossalai.zero.sharded_param.sharded_tensor import ShardedTensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from colossalai.zero.sharded_model._zero3_utils import get_shard
from colossalai.utils import get_current_device


class TensorListShardStrategy(BaseShardStrategy):
    """
    TensorListShardStrategy shards the tensor list into partitions in a greedy
    fashion. The tensor list is sorted by size and allocates each to tensor to
    the smallest partititon.

    This strategy is useful if we only want to partition gradients and optimizer
    states but not model parameters.
    """

    def __init__(self, process_group: Optional[dist.ProcessGroup] = None) -> None:
        super().__init__(process_group)

    def shard(self, tensor_list: List[Tensor]) -> List[Tensor]:
        """
        Sort and distribute the tensors into partitions. Each partition
        is owned by a rank.
        """
        params_per_rank = [[] for _ in range(self.world_size)]
        numel_per_rank = [0 for _ in range(self.world_size)]

        # partititon the parameters in a greedy fashion
        sorted_params = sorted(tensor_list, key=lambda x: x.numel(), reverse=True)
        for param in sorted_params:
            # allocate this parameter to the rank with
            # the smallest numel for load balancing purpose
            rank_to_go = numel_per_rank.index(min(numel_per_rank))
            params_per_rank[rank_to_go].append(param)
            numel_per_rank[rank_to_go] += param.numel()
        return params_per_rank[self.local_rank]

    def gather(self, tensor_list: List[ShardedTensor]) -> List[ShardedTensor]:
        # gather is not supported as we need to restore order of the original tensor list
        raise NotImplementedError("TensorListShardStrategy does not support gather")
