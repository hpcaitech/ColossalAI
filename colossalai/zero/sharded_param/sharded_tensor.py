import torch
import torch.distributed as dist
from colossalai.zero.sharded_model._zero3_utils import get_shard
from typing import Optional


class ShardedTensor(object):

    def __init__(self, tensor: torch.Tensor, process_group: Optional[dist.ProcessGroup] = None) -> None:
        r"""
        A tensor sharded in multiple processes.
        """
        self._payload = tensor
        self.process_group = process_group
        self.world_size = dist.get_world_size(self.process_group)
        self.local_rank = dist.get_rank(self.process_group)
        self._is_sharded = False
        self._payload = tensor

        self._origin_shape = tensor.shape
        self._origin_numel = tensor.numel()
        self._origin_dtype = tensor.dtype

    @property
    def is_sharded(self):
        return self._is_sharded

    @property
    def payload(self):
        return self._payload

    @payload.setter
    def payload(self, tensor):
        self._payload.copy_(tensor)

    @property
    def dtype(self):
        return self._origin_dtype

    @property
    def shape(self):
        return self._payload.shape

    def shard(self):
        if self._is_sharded:
            return
        self._payload, _ = get_shard(self._payload, self.local_rank, self.world_size)
        self._is_sharded = True

    def gather(self):
        if not self._is_sharded:
            return

        buffer_list = []
        payload_numel = self._payload.numel()
        for i in range(self.world_size):
            if i == self.local_rank:
                buffer_list.append(self._payload.cuda())
            else:
                buffer_list.append(torch.zeros(payload_numel).cuda())

        torch.distributed.all_gather(buffer_list,
                                     buffer_list[self.local_rank],
                                     group=self.process_group,
                                     async_op=False)
        self._payload = torch.narrow(torch.cat(buffer_list), 0, 0, self._origin_numel).view(self._origin_shape)
        self._is_sharded = False
