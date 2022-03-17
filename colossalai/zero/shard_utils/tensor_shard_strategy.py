from typing import List, Optional

import torch
import torch.distributed as dist

from colossalai.zero.shard_utils import BaseShardStrategy
from colossalai.zero.sharded_model._zero3_utils import get_shard
from colossalai.zero.sharded_param.sharded_tensor import ShardedTensor
from colossalai.utils import get_current_device


class TensorShardStrategy(BaseShardStrategy):

    def __init__(self, process_group: Optional[dist.ProcessGroup] = None) -> None:
        super().__init__(process_group)

    def shard(self, tensor_list: List[ShardedTensor]):
        for t in tensor_list:
            self._shard_tensor(t)

    def gather(self, tensor_list: List[ShardedTensor]):
        for t in tensor_list:
            self._gather_tensor(t)

    def _shard_tensor(self, t: ShardedTensor):
        if t.is_sharded:
            return
        sharded_payload, _ = get_shard(t.payload, self.local_rank, self.world_size)
        t.reset_payload(sharded_payload)
        t.is_sharded = True

    def _gather_tensor(self, t: ShardedTensor):
        if not t.is_sharded:
            return
        target_device = t.device
        buffer_list = []
        payload_numel = t.payload.numel()
        for i in range(self.world_size):
            if i == self.local_rank:
                buffer_list.append(t.payload.cuda(get_current_device()))
            else:
                buffer_list.append(torch.zeros(payload_numel, dtype=t.dtype, device=get_current_device()))

        torch.distributed.all_gather(buffer_list,
                                     buffer_list[self.local_rank],
                                     group=self.process_group,
                                     async_op=False)
        gathered_payload = torch.narrow(torch.cat(buffer_list), 0, 0, t.origin_numel).reshape(t.origin_shape)
        t.reset_payload(gathered_payload)
        t.to(target_device)
        t.is_sharded = False
