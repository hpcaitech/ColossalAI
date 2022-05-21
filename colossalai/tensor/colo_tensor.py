from .op_wrapper import _COLOSSAL_OPS
from copy import copy
import torch
from colossalai.tensor import TensorSpec
from .const import TensorType
from colossalai.tensor import distspec
from colossalai.tensor.dist_spec_mgr import DistSpecManager
from colossalai.tensor.distspec import _DistSpec
from torch.overrides import get_default_nowrap_functions


def _convert_output(output):
    if isinstance(output, torch.Tensor) and not isinstance(output, ColoTensor):
        output = ColoTensor.from_torch_tensor(output)
    elif isinstance(output, (list, tuple)):
        output = type(output)(_convert_output(o) for o in output)
    return output


class ColoTensor(torch.Tensor):
    """ Data Structure for Tensor in Colossal-AI
    1. It contains a torch.Tensor as an attribute.
    2. It supports lazy init the tensor's payload.
    3. It can hijack the torch functions which using ColoTensors as args to our customized functions.
    4. It supports distributing the tensor's payload to the shards among processes. (TODO)
    """

    def __new__(cls, data: torch.Tensor, spec: TensorSpec = TensorSpec(distspec.replicate())) -> 'ColoTensor':
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, data.requires_grad)

    def __init__(self, data: torch.Tensor, spec: TensorSpec = TensorSpec(distspec.replicate())) -> None:
        self._spec = copy(spec)
        self._type = TensorType.NONMODEL
        self._graph_node = None

    @property
    def spec(self) -> TensorSpec:
        return self._spec

    def set_spec(self, spec: TensorSpec) -> None:
        spec = copy(spec)
        self.convert_to_dist_spec_(spec.dist_spec)
        self._spec = spec

    def has_spec(self) -> bool:
        return self._spec.parallel_action is not None

    def is_model_data(self) -> bool:
        return self._type == TensorType.MODEL

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented
        global _COLOSSAL_OPS
        if func in _COLOSSAL_OPS:
            func = _COLOSSAL_OPS[func]

        with torch._C.DisableTorchFunction():
            ret = func(*args, **kwargs)
            if func in get_default_nowrap_functions():
                return ret
            else:
                return _convert_output(ret)

    def __repr__(self):
        return f'ColoTensor: {super().__repr__()}'

    def is_model_data(self) -> bool:
        return self._type == TensorType.MODEL

    def convert_to_dist_spec_(self, dist_spec: _DistSpec) -> None:
        with DistSpecManager.no_grad():
            self.data = DistSpecManager.handle_trans_spec(self, self.spec.dist_spec, dist_spec)
        self._spec.dist_spec = dist_spec

    def convert_to_dist_spec(self, dist_spec: _DistSpec) -> 'ColoTensor':
        spec = copy(self._spec)
        spec.dist_spec = dist_spec
        ret = DistSpecManager.handle_trans_spec(self, self.spec.dist_spec, dist_spec)
        return ColoTensor.from_torch_tensor(ret, spec)

    @staticmethod
    def from_torch_tensor(tensor: torch.Tensor, spec: TensorSpec = TensorSpec(distspec.replicate())) -> 'ColoTensor':
        tensor = tensor.as_subclass(ColoTensor)
        tensor.__init__(tensor, spec=spec)
        return tensor

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            with torch._C.DisableTorchFunction():
                data = self.data.clone()
            tensor = ColoTensor(data, spec=copy(self.spec))
            memo[id(self)] = tensor
            return tensor
