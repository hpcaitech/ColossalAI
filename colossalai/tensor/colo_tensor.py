from .op_wrapper import _COLOSSAL_OPS
from .const import TensorType
from copy import copy
import torch
from functools import lru_cache

from colossalai.tensor import ColoTensorSpec
from colossalai.tensor import distspec, ProcessGroup
from colossalai.tensor.dist_spec_mgr import DistSpecManager
from colossalai.tensor.distspec import _DistSpec, DistPlacementPattern
from typing import Optional, Set, Callable


@lru_cache(None)
def _get_my_nowrap_functions() -> Set[Callable]:
    Tensor = torch.Tensor
    return {
        Tensor._base.__get__,
        Tensor.grad.__get__,
        Tensor._grad.__get__,
        Tensor.data.__get__,  # make .data returns torch.Tensor rather than ColoTensor
    }


def _convert_output(output, pg: ProcessGroup):
    if type(output) == torch.Tensor:
        return ColoTensor.from_torch_tensor(output, ColoTensorSpec(pg))
    elif isinstance(output, (list, tuple)):
        return type(output)(_convert_output(o, pg) for o in output)
    else:
        return output


def _scan_for_pg_from_args(args, kwargs) -> ProcessGroup:
    for elem in args:
        if isinstance(elem, ColoTensor):
            pg = elem.get_process_group()
            return pg
        elif isinstance(elem, (list, tuple)):
            pg = _scan_for_pg_from_args(elem, {})
            if pg is not None:
                return pg
    for k, v in kwargs:
        if isinstance(v, ColoTensor):
            pg = v.get_process_group()
            return pg
    return None


class ColoTensor(torch.Tensor):
    """ Data Structure for Tensor in Colossal-AI. It is a subclass of torch.Tensor.
    Args:
        data (torch.Tensor): a torch tensor used as the payload the colotensor.
        spec (ColoTensorSpec, optional): the tensor spec of initialization. Defaults to ColoTensorSpec(distspec.replicate()).
    
    The signature of the function has to be consistent with the __new__ except for the 1st arg.
    The class should be initialized with a torch tensor in the following ways.
    1. directly init.
    >>> pg = ProcessGroup()
    >>> colo_t1 = ColoTensor(torch.randn(2,3), spec = ColoTensorSpec(pg, distspec.replicate())
    >>> # If initializaed in a shard model, the tensor passed in is one shard of the global tensor.
    >>> shard_spec = distspec.shard(process_group=ProcessGroup(tp=world_size), 
    >>>                 dims=[0], 
    >>>                 num_partitions=[world_size])
    >>> tensor_spec = ColoTensorSpec(pg, shard_spec)
    >>> colo_t2 = ColoTensor.from_torch_tensor(t_ref.clone(), tensor_spec)
    2. use static method from_torch_tensor
    >>> colo_t = ColoTensor.from_torch_tensor(torch.randn(2,3), spec = ColoTensorSpec(pg, distspec.replicate())
    """

    def __new__(cls, data: torch.Tensor, spec: ColoTensorSpec) -> 'ColoTensor':
        """__new__ 
        The signature of the __new__ has to be consistent with the torch.Tensor.
        Args:
            data (torch.Tensor): a torch tensor used as the payload the colotensor.
            spec (TensorSpec, optional): the tensor spec of initialization.
        Returns:
            ColoTensor: a ColoTensor wrappers the data.
        """
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, data.requires_grad)

    def __init__(self, data: torch.Tensor, spec: Optional[ColoTensorSpec] = None) -> None:
        # If not set spec, use a DP process group and replicate dist spec
        if spec is None:
            self.has_initialized = False
            self.dist_spec = distspec.replicate()
            self.compute_spec = None
            self.process_group = ProcessGroup()
        else:
            self.has_initialized = True
            self.dist_spec = spec.dist_attr
            self.compute_spec = spec.compute_attr
            if spec.pg is None:
                self.process_group = ProcessGroup()
            else:
                self.process_group = spec.pg

        self._type = TensorType.NONMODEL
        self._graph_node = None

    def has_compute_spec(self) -> bool:
        return self.compute_spec is not None

    def is_model_data(self) -> bool:
        return self._type == TensorType.MODEL

    def get_process_group(self) -> 'ProcessGroup':
        return self.process_group

    def set_process_group(self, pg: ProcessGroup):
        """set_process_group 
        change the pg of the ColoTensor. Note that the valid use cases is limited.
        Only existing pg is DP and dist spec is REPLICaTE is valid.
        Args:
            pg (ProcessGroup): target pg

        Raises:
            RuntimeError: 
            RuntimeError: 
        """
        assert isinstance(pg, ProcessGroup), f"pg as type {type(pg)} is invalid"
        if self.process_group.tp_world_size() != 1:
            raise RuntimeError("can not set_process_group on a ColoTensor whose process_group has tp world group")

        if self.dist_spec.placement.value != 'r':
            raise RuntimeError("can not set_process_group on a ColoTensor whose dist spec is not REPLICATE")

        self.process_group = pg

    def get_tp_world_size(self) -> int:
        return self.process_group.tp_world_size()

    def set_dist_spec(self, dist_spec: _DistSpec):
        """set_dist_spec 
        set dist spec and change the payloads.
        Args:
            dist_spec (_DistSpec): target dist spec.
        """
        assert isinstance(dist_spec, _DistSpec)
        assert self.process_group is not None
        self._convert_to_dist_spec(dist_spec)

    def set_tensor_spec(self, dist_spec, compute_spec):
        if dist_spec:
            assert isinstance(dist_spec, _DistSpec), f"{type(dist_spec)}"
            self.set_dist_spec(dist_spec)
        if compute_spec:
            self.compute_spec = compute_spec

    def has_compute_pattern(self, compute_pattern):
        return self.compute_spec.compute_pattern == compute_pattern

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
            if func in _get_my_nowrap_functions():
                return ret
            else:
                pg = _scan_for_pg_from_args(args, kwargs)
                return _convert_output(ret, pg)

    def __repr__(self):
        return f'ColoTensor:\n{super().__repr__()}\n{self.dist_spec}\n{self.process_group}'

    def _convert_to_dist_spec(self, dist_spec: _DistSpec) -> None:
        """_convert_to_dist_spec 
        Note the function will not handle the logic of backward propagation!
        It is used during model tensor initializations as an internal function.
        Args:
            dist_spec (_DistSpec): the target dist. spec.
        """
        assert self.grad_fn is None, "Current tensor has grad_fn and it can't get converted"
        with DistSpecManager.no_grad():
            self.data = DistSpecManager.handle_trans_spec(self.data, self.dist_spec, dist_spec, self.process_group)
        self.dist_spec = dist_spec

    def convert_to_dist_spec(self, dist_spec: _DistSpec) -> 'ColoTensor':
        ret = DistSpecManager.handle_trans_spec(self, self.dist_spec, dist_spec, self.process_group)
        return ColoTensor.from_torch_tensor(ret, ColoTensorSpec(self.process_group, dist_attr=dist_spec))

    def to_replicate_(self):
        """to_replicate_ 
        an inline member function, converting dist spec of the tensor to REPLICATE
        """
        self._convert_to_dist_spec(dist_spec=distspec.replicate())

    def to_replicate(self) -> 'ColoTensor':
        """to_replicate
        converting dist spec of the tensor to REPLICATE
        """
        return self.convert_to_dist_spec(distspec.replicate())

    @staticmethod
    def from_torch_tensor(tensor: torch.Tensor, spec: Optional[ColoTensorSpec] = None) -> 'ColoTensor':
        tensor = tensor.as_subclass(ColoTensor)
        tensor.__init__(tensor, spec=spec)
        return tensor

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            with torch._C.DisableTorchFunction():
                data = self.data.clone()
            tensor = ColoTensor(data, spec=copy(ColoTensorSpec(self.process_group, self.dist_spec, self.compute_spec)))
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
        if self.is_replicate():
            return super().view(*args)
        replicated_t = self.convert_to_dist_spec(dist_spec=distspec.replicate())
        return replicated_t.view(*args)

    def size_global(self, args: Optional[int] = None):
        """override the torch buildin size()
        the shape passed in must be in a replicate placement.
        Returns:
            ColoTensor: a tensor after viewed.
        """
        if self.is_replicate():
            if args is not None:
                return super().size(args)
            else:
                return super().size()

        spec = self.dist_spec
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

    # Some API for dist spec check

    def is_replicate(self):
        return self.dist_spec.placement == DistPlacementPattern.REPLICATE \
            or (len(self.dist_spec.num_partitions) == 1
                and self.dist_spec.num_partitions[0] == 1) \
            or (self.process_group.tp_world_size() == 1)

    def is_shard_1dcol(self):
        return self.dist_spec.placement == DistPlacementPattern.SHARD \
            and len(self.dist_spec.dims) == 1 and self.dist_spec.dims[0] == -1

    def is_shard_1drow(self):
        return self.dist_spec.placement == DistPlacementPattern.SHARD \
            and len(self.dist_spec.dims) == 1 and self.dist_spec.dims[0] == 0

    def is_sharded(self):
        return self.dist_spec.placement == DistPlacementPattern.SHARD