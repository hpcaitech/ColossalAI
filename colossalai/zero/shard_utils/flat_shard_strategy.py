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


class FlatShardStrategy(BaseShardStrategy):

    """
    FlatShardStategy flattens all tensors and shards the 1D tensor.
    """

    def __init__(self, process_group: Optional[dist.ProcessGroup] = None) -> None:
        super().__init__(process_group)

    def shard(self, tensor_list: List[Tensor]) -> List[Tensor]:
        flat_tensor = _flatten_dense_tensors(tensor_list)
        new_tensor_list = _unflatten_dense_tensors(flat_tensor, tensor_list)
        tensor_shape_list = [tensor.shape for tensor in tensor_list]

        # need to update the parameter data
        for old, new in zip(tensor_list, new_tensor_list):
            old.data = new.data

        shard, _ = get_shard(flat_tensor, rank=self.local_rank, world_size=self.world_size)
        shard.orig_shape_list = tensor_shape_list
        return [shard]

    def gather(self, tensor_list: List[ShardedTensor]) -> List[ShardedTensor]:
        assert len(tensor_list) == 1
        tensor = tensor_list[0]
        origin_tensor_shape_list = tensor.orig_shape_list

        tensor_shape = (tensor.shape[0] * self.world_size, ) + tensor.shape[1:]
        out = torch.empty(tensor_shape, dtype=tensor.dtype, device=get_current_device())
        out_list = list(torch.chunk(out, self.world_size, dim=0))
        dist.all_gather(out_list, tensor, group=self.process_group)

        gathered_flat_tensor = torch.cat(out_list, dim=0)

        dummy_tenosr_list = []
        for shape in origin_tensor_shape_list:
            dummy_tenosr_list.append(torch.zeros(shape, dtype=tensor.dtype, device=get_current_device()))
        gathered_tensor_list = _unflatten_dense_tensors(gathered_flat_tensor, dummy_tenosr_list)
        
        return gathered_tensor_list
