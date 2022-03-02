import torch
from colossalai.zero.sharded_model._zero3_utils import get_shard
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
import torch.distributed as dist
from typing import Union, Tuple, Optional


class ShardParam(object):
    r"""
    A wrapper to torch.nn.Parameter. Shard a param
    on memory space of different processes.
    """

    def __init__(self,
                 param: Union[torch.nn.Parameter, Tuple[int, ...]],
                 process_group: Optional[dist.ProcessGroup] = None,
                 is_sharded: bool = False,
                 device: Optional[torch.device] = None) -> None:
        r"""
        param: either an existing torch parameter or a tuple, indicate allocate a new param with the tuple as shape.
        process_group: the process group storing the shared data.
        is_sharded: is shared the param during __init__.
        device: the device to place param data payload on
        """
        self.process_group = process_group or gpc.get_group(ParallelMode.DATA)
        self.world_size = dist.get_world_size(self.process_group)
        self.local_rank = dist.get_rank(self.process_group)
        self.is_sharded = False

        # Hijack the data payload of param
        if isinstance(param, torch.nn.Parameter):
            self._param_payload = param.data.to(device)
            self._origin_shape = param.shape
            self._origin_numel = param.numel()
            if is_sharded:
                self.shard()
        elif isinstance(param, tuple):
            self._origin_shape = param.shape
            self._origin_numel = param.numel()

            # TODO(jiaruifang) can be optimized. Directly allocate payload as the sharded shape.
            assert device is not None, "You have to assign a device to initialize a ShardParam from a shape tuple"
            self._param_payload = torch.empty(self._origin_shape, device=device)
            if is_sharded:
                self.shard()
        else:
            raise RuntimeError(f"Initialize ShardParam failed. The 2nd parameter is wrong type {type(param)}")

        self._payload_numel = None

    def payload(self, target_device: torch.device):
        r"""
        get the payload and move it to target device
        """
        return self._param_payload.to(target_device)

    def shard(self):
        r"""
        Distributed the payload of param to all processes.
        """
        if self.is_sharded:
            return
        self._param_payload, _ = get_shard(self._param_payload, self.local_rank, self.world_size)
        self.is_sharded = True

    def gather(self):
        r"""
        Collect the payload of param from different processes to process of local rank.
        The payload has to be moved to cuda memory before communication.
        """
        if not self.is_sharded:
            return

        buffer_list = []
        payload_numel = self._param_payload.numel()
        for i in range(self.world_size):
            if i == self.local_rank:
                buffer_list.append(self._param_payload.cuda())
            else:
                buffer_list.append(torch.zeros(payload_numel).cuda())

        torch.distributed.all_gather(buffer_list,
                                     buffer_list[self.local_rank],
                                     group=self.process_group,
                                     async_op=False)
        self._param_payload = torch.narrow(torch.cat(buffer_list), 0, 0, self._origin_numel).view(self._origin_shape)
        self.is_sharded = False
