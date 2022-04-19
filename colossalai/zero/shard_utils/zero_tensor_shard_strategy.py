from typing import Optional

import torch
import torch.distributed as dist
from colossalai.utils import get_current_device
from colossalai.zero.sharded_param.tensor_utils import colo_model_data_tensor_move_inline
from colossalai.zero.sharded_param.sharded_tensor import ShardedTensor
from colossalai.zero.comm import ZeroDist

from .tensor_shard_strategy import TensorShardStrategy


class ZeroTensorShardStrategy(TensorShardStrategy):
    """Use the same shard scheme as `TensorShardStrategy`'s.
    But its all-gather operation is in-place, meaning that no extra buffer is created.
    Extra buffer is created when using `torch.distributed.all_gather`.
    This can reduce peak memory used in zero-offload.
    You should notice that this strategy is highly coupled with zero.
    You can not change its communication group and must use ZeroContext to create your model.
    """

    def _gather_tensor(self, t: ShardedTensor, process_group: Optional[dist.ProcessGroup] = None):
        if not t.is_sharded:
            return
        target_device = t.device
        payload_numel = t.payload.numel()
        world_size = dist.get_world_size(process_group)
        rank = dist.get_rank(process_group)

        buffer = torch.empty(payload_numel * world_size, dtype=t.payload.dtype, device=get_current_device())
        buffer_list = list(torch.chunk(buffer, chunks=world_size, dim=0))
        buffer_list[rank].copy_(t.payload)

        ZeroDist.zero_all_gather(buffer)    # notice: process_group is useless here
        gathered_payload = torch.narrow(buffer, 0, 0, t.origin_numel).reshape(t.origin_shape)
        t.reset_payload(gathered_payload)
        colo_model_data_tensor_move_inline(t, target_device)
        t.is_sharded = False
