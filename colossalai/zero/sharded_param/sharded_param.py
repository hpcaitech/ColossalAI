import numpy
import torch
import torch.distributed as dist
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.zero.sharded_model._zero3_utils import get_shard
from colossalai.zero.sharded_param import ShardedTensor
from typing import Union, Tuple, Optional


class ShardedParamV2(object):

    def __init__(self,
                 param: torch.nn.Parameter,
                 process_group: Optional[dist.ProcessGroup] = None,
                 rm_torch_payload=False) -> None:
        self._data_sharded_tensor = ShardedTensor(param.data, process_group)
        if param.requires_grad and param.grad is not None:
            self._grad_sharded_tensor = ShardedTensor(param.grad, process_group)
            param.grad = None
        else:
            self._grad_sharded_tensor = None

        # make sure the shared param is the only owner of payload
        # The param.data maybe used to init the other part of the model.
        # For example: File "resnet.py", line 190, in __init__
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # So we can not empty the .data at this time
        self.param = param
        if rm_torch_payload:
            self.remove_torch_payload()

    def remove_torch_payload(self):
        self.param.data = torch.empty([], dtype=self.param.dtype, device=self.param.device)

    @property
    def data(self):
        return self._data_sharded_tensor.payload

    @data.setter
    def data(self, t: torch.Tensor):
        self._data_sharded_tensor.payload = t

    @property
    def grad(self):
        if self._grad_sharded_tensor:
            return self._grad_sharded_tensor.payload
        else:
            return None

    @grad.setter
    def grad(self, t: torch.Tensor):
        self._grad_sharded_tensor.payload = t


class ShardedParam(object):
    r"""
    A wrapper to torch.nn.Parameter. Shard a param
    on memory space of different processes.
    """

    def __init__(self,
                 other: Union[torch.nn.Parameter, Tuple[int, ...]],
                 process_group: Optional[dist.ProcessGroup] = None,
                 is_sharded: bool = False,
                 device: Optional[torch.device] = None) -> None:
        r"""
        other: either an existing torch parameter or a tuple, indicate allocate a new param with the tuple as shape.
        process_group: the process group storing the shared data.
        is_sharded: is shared the param during __init__.
        device: the device to place param data payload on
        """
        self.process_group = process_group or gpc.get_group(ParallelMode.DATA)
        self.world_size = dist.get_world_size(self.process_group)
        self.local_rank = dist.get_rank(self.process_group)
        self.is_sharded = False
        self.device = device

        # Hijack the data payload of param
        if isinstance(other, torch.nn.Parameter):
            self._param_payload = other.data.to(device)
            self._origin_shape = other.shape
            self._origin_numel = other.numel()
            if is_sharded:
                self.shard()
        elif isinstance(other, tuple):
            self._origin_shape = other
            self._origin_numel = numpy.prod(other)

            # TODO(jiaruifang) can be optimized. Directly allocate payload as the sharded shape.
            assert device is not None, "You have to assign a device to initialize a ShardParam from a shape tuple"
            self._param_payload = torch.empty(self._origin_shape, device=device)
            if is_sharded:
                self.shard()
        else:
            raise RuntimeError(f"Initialize ShardParam failed. The 2nd parameter is wrong type {type(other)}")

        self._payload_numel = None

    def payload(self, target_device: Optional[torch.device] = None):
        r"""
        get the payload and move it to target device
        """
        if target_device is not None:
            return self._param_payload.to(target_device)
        return self._param_payload

    def set_payload(self, data: torch.Tensor):
        r"""
        set payload as data
        """
        assert self._param_payload.shape == data.shape
        self._param_payload.copy_(data)

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

    @property
    def origin_dtype(self):
        return self._origin_dtype
