from typing import List, Optional

import torch
import torch.distributed as dist

from colossalai.accelerator import get_accelerator
from colossalai.legacy.zero.gemini.tensor_utils import colo_model_data_tensor_move_inline
from colossalai.legacy.zero.shard_utils import BaseShardStrategy
from colossalai.legacy.zero.shard_utils.commons import get_shard
from colossalai.legacy.zero.sharded_param.sharded_tensor import ShardedTensor


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
        """Shard tensor among processes.

        Args:
            t (ShardedTensor): a tensor to be sharded.
            process_group (Optional[dist.ProcessGroup], optional): the process group among which tensor shards.
            Defaults to None.
        """
        if t.is_sharded:
            return
        if t.payload.device.type == "cuda":
            assert t.payload.device == get_accelerator().get_current_device(), (
                f"shard tensor on cuda device index {t.payload.device.index},"
                f" but current cuda device is {get_accelerator().get_current_device()}"
            )
        sharded_payload, _ = get_shard(t.payload, dist.get_rank(process_group), dist.get_world_size(process_group))
        t.payload_reset(sharded_payload)
        t.is_sharded = True

    def _gather_tensor(self, t: ShardedTensor, process_group: Optional[dist.ProcessGroup] = None):
        if not t.is_sharded:
            return
        target_device = t.device
        payload_numel = t.payload.numel()
        world_size = dist.get_world_size(process_group)
        rank = dist.get_rank(process_group)

        buffer = torch.empty(
            payload_numel * world_size, dtype=t.payload.dtype, device=get_accelerator().get_current_device()
        )
        buffer_list = list(torch.chunk(buffer, chunks=world_size, dim=0))
        buffer_list[rank].copy_(t.payload)

        dist.all_gather(buffer_list, buffer_list[rank], group=process_group, async_op=False)
        gathered_payload = torch.narrow(buffer, 0, 0, t.origin_numel).reshape(t.origin_shape)
        t.payload_reset(gathered_payload)
        colo_model_data_tensor_move_inline(t, target_device)
        t.is_sharded = False
