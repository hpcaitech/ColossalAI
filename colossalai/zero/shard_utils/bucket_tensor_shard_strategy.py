from typing import List, Optional

import torch
import torch.distributed as dist
from colossalai.utils import get_current_device
from colossalai.zero.sharded_param.sharded_tensor import ShardedTensor
from torch._utils import _flatten_dense_tensors as flatten

from .tensor_shard_strategy import TensorShardStrategy


class BucketTensorShardStrategy(TensorShardStrategy):
    """Use the same shard scheme as `TensorShardStrategy`'s, but it gathers tensors of a sub-module together, 
    which will fully utilize network bandwidth. 
    It is especially useful when sub-module contains bias, 
    since we cannot utilize network bandwidth well if we only gather a bias tensor (bias is usaully small).
    """

    def gather(self, tensor_list: List[ShardedTensor], process_group: Optional[dist.ProcessGroup] = None):

        tensor_list: List[ShardedTensor] = [t for t in tensor_list if t.is_sharded]
        if len(tensor_list) == 0:
            return
        target_device = tensor_list[0].device
        dtype = tensor_list[0].dtype
        buffer_list: List[torch.Tensor] = []
        tensor_numels = [t.payload.numel() for t in tensor_list]
        buffer_size = sum(tensor_numels)
        world_size = dist.get_world_size(process_group)
        rank = dist.get_rank(process_group)
        for i in range(world_size):
            if i == rank:
                buffer_list.append(flatten([t.payload for t in tensor_list]).cuda(get_current_device()))
            else:
                buffer_list.append(torch.zeros(buffer_size, dtype=dtype, device=get_current_device()))
        dist.all_gather(buffer_list, buffer_list[rank], group=process_group)
        # Move to target device before splitting buffer
        # Ensure we utilize maximum PCIE bandwidth
        buffer_list = [buffer.to(target_device) for buffer in buffer_list]
        offset = 0
        for i, t in enumerate(tensor_list):
            gathered_payload = [buffer[offset:offset + tensor_numels[i]] for buffer in buffer_list]
            gathered_payload = torch.cat(gathered_payload)[:t.origin_numel].view(t.origin_shape)
            t.payload_reset(gathered_payload)
            t.is_sharded = False
            offset += tensor_numels[i]
