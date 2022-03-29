import torch
import torch.distributed as dist
from colossalai.zero.sharded_param import ShardedTensor
from typing import Optional, Tuple
from colossalai.utils.memory_utils.utils import colo_tensor_mem_usage


class ShardedParamV2(object):

    def __init__(self,
                 param: torch.nn.Parameter,
                 process_group: Optional[dist.ProcessGroup] = None,
                 rm_torch_payload=False) -> None:
        self._sharded_data_tensor: ShardedTensor = ShardedTensor(param.data, process_group)
        self.fp16_grad: Optional[torch.Tensor] = None
        self.fp32_grad: Optional[torch.Tensor] = None
        # This attribute must be initialized in ShardedModel
        self.offload_grad: bool = False

        # make sure the shared param is the only owner of payload
        # The param.data maybe used to init the other part of the model.
        # For example: File "resnet.py", line 190, in __init__
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # So we can not empty the .data at this time
        self.param = param
        if rm_torch_payload:
            self.remove_torch_payload()

        # Backward count for handle local grad accumulation
        # This value will increment by 1 in every pre-bwd hook
        # And will be reset to 0 in every final-bwd hook
        self.bwd_count = 0

    def remove_torch_payload(self):
        self.param.data = torch.empty([], dtype=self.param.dtype, device=self.param.device)

    @property
    def sharded_data_tensor(self):
        return self._sharded_data_tensor

    @property
    def param_is_sharded(self):
        return self._sharded_data_tensor.is_sharded

    def get_memory_usage(self) -> Tuple[int, int]:
        """
        get the memory usage of the param, including data and grad
        Returns:
            Tuple[int, int]: cuda mem usage in Byte, cpu memory usage in Byte
        """
        cuda_mem_use, cpu_mem_use = 0, 0

        def _update_mem_use(t: Optional[torch.Tensor]):
            if t is None:
                return
            assert isinstance(t, torch.Tensor)
            nonlocal cuda_mem_use
            nonlocal cpu_mem_use
            t_cuda, t_cpu = colo_tensor_mem_usage(t)
            cuda_mem_use += t_cuda
            cpu_mem_use += t_cpu

        address_set = set()
        _update_mem_use(self.sharded_data_tensor.payload)
        address_set.add(self.sharded_data_tensor.payload.data_ptr())

        if self.fp16_grad is not None and self.fp16_grad.data_ptr() not in address_set:
            _update_mem_use(self.fp16_grad)
            address_set.add(self.fp16_grad.data_ptr())

        if self.fp32_grad is not None and self.fp32_grad.data_ptr() not in address_set:
            _update_mem_use(self.fp32_grad)
            address_set.add(self.fp32_grad.data_ptr())

        if self.param.data is not None and self.param.data.data_ptr() not in address_set:
            _update_mem_use(self.param.data)
            address_set.add(self.param.data.data_ptr())

        if self.param.grad is not None and self.param.grad.data_ptr() not in address_set:
            _update_mem_use(self.param.grad)
            address_set.add(self.param.grad.data_ptr())

        return cuda_mem_use, cpu_mem_use
