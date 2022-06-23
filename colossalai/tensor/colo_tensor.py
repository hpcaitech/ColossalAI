from .op_wrapper import _COLOSSAL_OPS
from .const import TensorType
from copy import copy
import torch
from torch.overrides import get_default_nowrap_functions

from colossalai.tensor import TensorSpec
from colossalai.tensor import distspec
from colossalai.tensor.dist_spec_mgr import DistSpecManager
from colossalai.tensor.distspec import _DistSpec


def _convert_output(output):
    if isinstance(output, torch.Tensor) and not isinstance(output, ColoTensor):
        output = ColoTensor.from_torch_tensor(output)
    elif isinstance(output, (list, tuple)):
        output = type(output)(_convert_output(o) for o in output)
    return output


class ColoTensor(torch.Tensor):
    """ Data Structure for Tensor in Colossal-AI. It is a subclass of torch.Tensor.
    Args:
        data (torch.Tensor): a torch tensor used as the payload the colotensor.
        spec (TensorSpec, optional): the tensor spec of initialization. Defaults to TensorSpec(distspec.replicate()).
    
    The signature of the function has to be consistent with the __new__ except for the 1st arg.
    The class should be initialized with a torch tensor in the following ways.
    1. directly init.
    >>> colo_t1 = ColoTensor(torch.randn(2,3), spec = TensorSpec(distspec.replicate())
    >>> # If initializaed in a shard model, the tensor passed in is one shard of the global tensor.
    >>> shard_spec = distspec.shard(process_group=gpc.get_group(ParallelMode.DATA), 
    >>>                 dims=[0], 
    >>>                 num_partitions=[world_size])
    >>> tensor_spec = TensorSpec(shard_spec)
    >>> colo_t2 = ColoTensor.from_torch_tensor(t_ref.clone(), tensor_spec)
    2. use static method from_torch_tensor
    >>> colo_t = ColoTensor.from_torch_tensor(torch.randn(2,3), spec = TensorSpec(distspec.replicate())
    """

    def __new__(cls, data: torch.Tensor, spec: TensorSpec = TensorSpec(distspec.replicate())) -> 'ColoTensor':
        """__new__ 
        The signature of the __new__ has to be consistent with the torch.Tensor.
        Args:
            data (torch.Tensor): a torch tensor used as the payload the colotensor.
            spec (TensorSpec, optional): the tensor spec of initialization. Defaults to TensorSpec(distspec.replicate())
        Returns:
            ColoTensor: a ColoTensor wrappers the data.
        """
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, data.requires_grad)

    def __init__(self, data: torch.Tensor, spec: TensorSpec = TensorSpec(distspec.replicate())) -> None:
        self._tensor_spec = copy(spec)
        self._type = TensorType.NONMODEL
        self._graph_node = None

    @property
    def spec(self) -> TensorSpec:
        return self._tensor_spec

    def set_spec(self, spec: TensorSpec) -> None:
        spec = copy(spec)
        self._convert_to_dist_spec(spec.dist_spec)
        self._tensor_spec = spec

    def has_spec(self) -> bool:
        return self._tensor_spec.parallel_action is not None

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

    def _convert_to_dist_spec(self, dist_spec: _DistSpec) -> None:
        """_convert_to_dist_spec 
        Note the function will not handle the logic of backward propagation!
        It is used during model tensor initializations as an internal function.
        Args:
            dist_spec (_DistSpec): the target dist. spec.
        """
        with DistSpecManager.no_grad():
            self.data = DistSpecManager.handle_trans_spec(self, self.spec.dist_spec, dist_spec)
        self._tensor_spec.dist_spec = dist_spec

    def convert_to_dist_spec(self, dist_spec: _DistSpec) -> 'ColoTensor':
        tensor_spec = copy(self._tensor_spec)
        tensor_spec.dist_spec = dist_spec
        ret = DistSpecManager.handle_trans_spec(self, self.spec.dist_spec, dist_spec)
        return ColoTensor.from_torch_tensor(ret, tensor_spec)

    def to_replicate_(self):
        """to_replicate_ 
        an inline member function, converting dist spec of the tensor to REPLICATE
        """
        self.data = DistSpecManager.handle_trans_spec(self, self.spec.dist_spec, distspec.replicate())
        self._tensor_spec.dist_spec = distspec.replicate()

    def to_replicate(self) -> 'ColoTensor':
        """to_replicate
        converting dist spec of the tensor to REPLICATE
        """
        return self.convert_to_dist_spec(distspec.replicate(self.spec.get_process_group()))

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

    # TODO(jiaruifang) a patch for gpt test.
    # We need to override the member function must operate on a replicated tensor
    # def view(self, *args, **kwargs):
    #     self.data = DistSpecManager.handle_trans_spec(self,
    #                 self.spec.dist_spec,
    #                 distspec.replicate(self.spec.get_process_group()))
    #     # self._tensor_spec.dist_spec = distspec.replicate(self.spec.get_process_group())
    #     self.data.view(*args, **kwargs)
    #     return ColoTensor.from_torch_tensor(self.data)
