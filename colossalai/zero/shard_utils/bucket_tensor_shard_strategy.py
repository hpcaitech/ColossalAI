from typing import List

import torch
import torch.distributed as dist
from colossalai.utils import get_current_device
from colossalai.zero.sharded_param.sharded_tensor import ShardedTensor
from torch._utils import _flatten_dense_tensors as flatten

from .tensor_shard_strategy import TensorShardStrategy


class BucketTensorShardStrategy(TensorShardStrategy):

    def gather(self, tensor_list: List[ShardedTensor]):
        tensor_list: List[ShardedTensor] = [t for t in tensor_list if t.is_sharded]
        if len(tensor_list) == 0:
            return
        target_device = tensor_list[0].device
        dtype = tensor_list[0].dtype
        buffer_list: List[torch.Tensor] = []
        tensor_numels = [t.payload.numel() for t in tensor_list]
        buffer_size = sum(tensor_numels)
        for i in range(self.world_size):
            if i == self.local_rank:
                buffer_list.append(flatten([t.payload for t in tensor_list]).cuda(get_current_device()))
                # Release payload here, to decrease peak memory usage
                for t in tensor_list:
                    t.reset_payload(None)
            else:
                buffer_list.append(torch.zeros(buffer_size, dtype=dtype, device=get_current_device()))
        dist.all_gather(buffer_list, buffer_list[self.local_rank], group=self.process_group)
        # Move to target device before splitting buffer
        # Ensure we utilize maximum PCIE bandwidth
        buffer_list = [buffer.to(target_device) for buffer in buffer_list]
        offset = 0
        for i, t in enumerate(tensor_list):
            gathered_payload = [buffer[offset:offset + tensor_numels[i]] for buffer in buffer_list]
            gathered_payload = torch.cat(gathered_payload)[:t.origin_numel].view(t.origin_shape)
            t.reset_payload(gathered_payload)
            t.is_sharded = False
            offset += tensor_numels[i]
