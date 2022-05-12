import torch
from typing import Optional, Tuple
from colossalai.zero.sharded_param.sharded_tensor import ShardedTensor
from colossalai.gemini.tensor_utils import colo_tensor_mem_usage
from colossalai.gemini.stateful_tensor import StatefulTensor, TensorState
from typing import List

EMPTY_TENSOR_DICT = {}


def get_empty_tensor(device: torch.device, dtype: torch.dtype):
    key = (device, dtype)
    if key not in EMPTY_TENSOR_DICT:
        EMPTY_TENSOR_DICT[key] = torch.empty(0, dtype=dtype, device=device)

    return EMPTY_TENSOR_DICT[key]


class ShardedParamV2(object):

    def __init__(self, param: torch.nn.Parameter, set_data_none: bool = False) -> None:
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
        if set_data_none:
            self.set_data_none()

    def get_payload_tensors(self) -> List[StatefulTensor]:
        """returns stateful tensors kept by this class.
        """
        return [self._sharded_data_tensor]

    def set_data_none(self):
        self.param.data = get_empty_tensor(self.sharded_data_tensor.device, self.sharded_data_tensor.dtype)

    def set_grad_none(self):
        self.saved_grad.set_null()

    @property
    def sharded_data_tensor(self):
        return self._sharded_data_tensor

    @property
    def data_payload(self):
        assert not self.sharded_data_tensor.is_null()
        return self.sharded_data_tensor.payload

    @property
    def grad_payload(self):
        assert not self.saved_grad.is_null()
        return self.saved_grad.payload

    @property
    def param_is_sharded(self):
        return self.sharded_data_tensor.is_sharded

    def data_payload_reset(self, tensor: torch.Tensor):
        assert type(tensor) is torch.Tensor
        assert tensor.requires_grad is False
        self.sharded_data_tensor.payload_reset(tensor)

    def grad_payload_reset(self, tensor: torch.Tensor):
        assert type(tensor) is torch.Tensor
        assert tensor.requires_grad is False
        self.saved_grad.payload_reset(tensor)

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
        _update_mem_use(self.data_payload)
        address_set.add(self.data_payload.data_ptr())

        if not self.saved_grad.is_null() and self.saved_grad.data_ptr() not in address_set:
            _update_mem_use(self.grad_payload)
            address_set.add(self.saved_grad.data_ptr())

        if self.param.data is not None and self.param.data.data_ptr() not in address_set:
            _update_mem_use(self.param.data)
            address_set.add(self.param.data.data_ptr())

        if self.param.grad is not None and self.param.grad.data_ptr() not in address_set:
            _update_mem_use(self.param.grad)

        return cuda_mem_use, cpu_mem_use
