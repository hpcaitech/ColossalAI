from typing import List, Optional

import torch
import torch.distributed as dist
from colossalai.utils import get_current_device
from colossalai.utils.memory_utils.utils import colo_model_data_tensor_move_inline
from colossalai.zero.shard_utils import BaseShardStrategy
from colossalai.zero.shard_utils.commons import get_shard
from colossalai.zero.sharded_param.sharded_tensor import ShardedTensor


class TensorShardStrategy(BaseShardStrategy):
    """
    A naive implementation which shard each tensor evenly over all ranks
    """

    def shard(self, tensor_list: List[ShardedTensor], process_group: Optional[dist.ProcessGroup] = None):
        for t in tensor_list:
            self._shard_tensor(t, process_group)

    def gather(self, tensor_list: List[ShardedTensor], process_group: Optional[dist.ProcessGroup] = None):
        for t in tensor_list:
            self._gather_tensor(t, process_group)

    def _shard_tensor(self, t: ShardedTensor, process_group: Optional[dist.ProcessGroup] = None):
        """ Shard tensor among processes.

        Args:
            t (ShardedTensor): a tensor to be sharded.
            process_group (Optional[dist.ProcessGroup], optional): the process group among which tensor shards. 
            Defaults to None.
        """
        if t.is_sharded:
            return
        if t.payload.device.type == 'cuda':
            assert t.payload.device.index == get_current_device(), f"shard tensor on cuda device index {t.payload.device.index},"\
                f" but current cuda device is {get_current_device()}"
        sharded_payload, _ = get_shard(t.payload, dist.get_rank(process_group), dist.get_world_size(process_group))
        t.reset_payload(sharded_payload)
        t.is_sharded = True

    def _gather_tensor(self, t: ShardedTensor, process_group: Optional[dist.ProcessGroup] = None):
        if not t.is_sharded:
            return
        target_device = t.device
        buffer_list = []
        payload_numel = t.payload.numel()
        world_size = dist.get_world_size(process_group)
        rank = dist.get_rank(process_group)
        for i in range(world_size):
            if i == rank:
                buffer_list.append(t.payload.cuda(get_current_device()))
            else:
                buffer_list.append(torch.zeros(payload_numel, dtype=t.dtype, device=get_current_device()))

        dist.all_gather(buffer_list, buffer_list[rank], group=process_group, async_op=False)
        gathered_payload = torch.narrow(torch.cat(buffer_list), 0, 0, t.origin_numel).reshape(t.origin_shape)
        t.reset_payload(gathered_payload)
        colo_model_data_tensor_move_inline(t, target_device)
        t.is_sharded = False
