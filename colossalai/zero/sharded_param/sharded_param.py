import torch
from colossalai.zero.sharded_param import ShardedTensor
from typing import Optional, Tuple
from colossalai.zero.shard_utils.tensor_utils import colo_tensor_mem_usage
from .tensorful_state import StatefulTensor, TensorState
from typing import List

# use this tensor as empty data point for parameters
# we do not want users use param.data when its torch payload is removed
# empty tensor is expected to raise error when get used
FAKE_EMPTY_TENSOR = torch.BoolTensor([], device='cpu')


class ShardedParamV2(object):

    def __init__(self, param: torch.nn.Parameter, rm_torch_payload=False) -> None:
        self._sharded_data_tensor: ShardedTensor = ShardedTensor(param.data)
        self.saved_grad: StatefulTensor = StatefulTensor(None, TensorState.FREE)
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

    def get_payload_tensors(self) -> List[StatefulTensor]:
        """returns stateful tensors kept by this class.
        """
        return [self._sharded_data_tensor]

    def remove_torch_payload(self):
        self.param.data = FAKE_EMPTY_TENSOR.to(self._sharded_data_tensor.device, self._sharded_data_tensor.dtype)

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

        if not self.saved_grad.is_null() and self.saved_grad.data_ptr() not in address_set:
            _update_mem_use(self.saved_grad.payload)
            address_set.add(self.saved_grad.data_ptr())

        if self.param.data is not None and self.param.data.data_ptr() not in address_set:
            _update_mem_use(self.param.data)
            address_set.add(self.param.data.data_ptr())

        if self.param.grad is not None and self.param.grad.data_ptr() not in address_set:
            _update_mem_use(self.param.grad)
            address_set.add(self.param.grad.data_ptr())

        return cuda_mem_use, cpu_mem_use
