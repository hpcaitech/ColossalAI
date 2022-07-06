from .op_wrapper import _COLOSSAL_OPS
from .const import TensorType
from copy import copy
import torch
from torch.overrides import get_default_nowrap_functions

from colossalai.tensor import TensorSpec
from colossalai.tensor import distspec
from colossalai.tensor.dist_spec_mgr import DistSpecManager
from colossalai.tensor.distspec import _DistSpec
from typing import Optional


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
    >>> shard_spec = distspec.shard(process_group=ProcessGroup(tp=world_size), 
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
    def tensor_spec(self) -> TensorSpec:
        return self._tensor_spec

    @tensor_spec.setter
    def tensor_spec(self, tenseor_spec: TensorSpec):
        spec = copy(spec)
        self._convert_to_dist_spec(spec.dist_spec)
        self._tensor_spec = spec

    def set_tensor_spec(self, spec: TensorSpec) -> None:
        spec = copy(spec)
        self._convert_to_dist_spec(spec.dist_spec)
        self._tensor_spec = spec

    def has_compute_spec(self) -> bool:
        return self._tensor_spec.compute_spec is not None

    def is_model_data(self) -> bool:
        return self._type == TensorType.MODEL

    def get_process_group(self) -> 'ProcessGroup':
        return self._tensor_spec.dist_spec.process_group

    def get_tp_world_size(self) -> int:
        return self._tensor_spec.dist_spec.process_group.tp_world_size()

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
            self.data = DistSpecManager.handle_trans_spec(self, self.tensor_spec.dist_spec, dist_spec)
        self._tensor_spec.dist_spec = dist_spec

    def convert_to_dist_spec(self, dist_spec: _DistSpec) -> 'ColoTensor':
        tensor_spec = copy(self._tensor_spec)
        tensor_spec.dist_spec = dist_spec
        ret = DistSpecManager.handle_trans_spec(self, self.tensor_spec.dist_spec, dist_spec)
        return ColoTensor.from_torch_tensor(ret, tensor_spec)

    def to_replicate_(self):
        """to_replicate_ 
        an inline member function, converting dist spec of the tensor to REPLICATE
        """
        self.data = DistSpecManager.handle_trans_spec(self, self.tensor_spec.dist_spec, distspec.replicate())
        self._tensor_spec.dist_spec = distspec.replicate()

    def to_replicate(self) -> 'ColoTensor':
        """to_replicate
        converting dist spec of the tensor to REPLICATE
        """
        return self.convert_to_dist_spec(distspec.replicate(self.tensor_spec.get_process_group()))

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
            tensor = ColoTensor(data, spec=copy(self.tensor_spec))
            memo[id(self)] = tensor
            return tensor

    ##### override builtin functions which must use tensor in replicate placement ####

    def view_local(self, *args) -> 'ColoTensor':
        return super().view(*args)

    def size_local(self, *args, **kwargs) -> torch.Size:
        return super().size(*args, **kwargs)

    def view_global(self, *args) -> 'ColoTensor':
        """override the torch buildin view()
        the args passed in must be in a replicate placement.
        Returns:
            ColoTensor: a tensor after viewed.
        """
        if self.tensor_spec.is_replicate():
            return super().view(*args)
        # TODO(jiaruifang) check why this not work
        # self.data = self.to_replicate()
        self.data = DistSpecManager.handle_trans_spec(self.data, self.tensor_spec.dist_spec, distspec.replicate())
        self._tensor_spec.dist_spec = distspec.replicate()
        return super().view(*args)

    def size_global(self, args: Optional[int] = None):
        """override the torch buildin size()
        the shape passed in must be in a replicate placement.
        Returns:
            ColoTensor: a tensor after viewed.
        """
        if self.tensor_spec.is_replicate():
            if args is not None:
                return super().size(args)
            else:
                return super().size()

        spec = self.tensor_spec.dist_spec
        dims = spec.dims
        num_partitions = spec.num_partitions
        # import inspect
        # print(*['{:40}| {}:{}\n'.format(x.function, x.filename, x.lineno) for x in inspect.stack()])

        size_list = list(super().size())
        for dim, num_partition in zip(dims, num_partitions):
            size_list[dim] *= num_partition
        if args is not None:
            return size_list[args]
        else:
            return torch.Size(size_list)
