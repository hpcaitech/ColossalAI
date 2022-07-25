from .op_wrapper import _COLOSSAL_OPS
from .const import TensorType
from copy import copy
import torch
from functools import lru_cache

from colossalai.tensor import ColoTensorSpec
from colossalai.tensor import ProcessGroup, ReplicaSpec
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


def _convert_output(output, colo_spec: ColoTensorSpec):
    if type(output) == torch.Tensor:
        return ColoTensor.from_torch_tensor(output, colo_spec)
    elif isinstance(output, (list, tuple)):
        return type(output)(_convert_output(o, colo_spec) for o in output)
    else:
        return output


def _get_spec_from_args(args, kwargs) -> ColoTensorSpec:
    for elem in args:
        if isinstance(elem, ColoTensor):
            pg = elem.get_process_group()
            dp = elem.dist_spec
            return ColoTensorSpec(pg, dp)
        elif isinstance(elem, (list, tuple)):
            spec = _get_spec_from_args(elem, {})
            if spec is not None:
                return spec
    for k, v in kwargs.items():
        if isinstance(v, ColoTensor):
            pg = v.get_process_group()
            dp = v.dist_spec
            return ColoTensorSpec(pg, dp)
    return None


class ColoTensor(torch.Tensor):
    """ Data Structure for Tensor in Colossal-AI. It is a subclass of torch.Tensor.
    Args:
        data (torch.Tensor): a torch tensor used as the payload the colotensor.
        spec (ColoTensorSpec, optional): the tensor spec of initialization. Defaults to ColoTensorSpec(ReplicaSpec()).

    The signature of the function has to be consistent with the __new__ except for the 1st arg.
    The class should be initialized with a torch tensor in the following ways.
    1. directly init.
    >>> pg = ProcessGroup()
    >>> colo_t1 = ColoTensor(torch.randn(2,3), spec = ColoTensorSpec(pg, ReplicaSpec())
    >>> # If initializaed in a shard model, the tensor passed in is one shard of the global tensor.
    >>> shard_spec = ShardSpec(process_group=ProcessGroup(tp=world_size), 
    >>>                 dims=[0], 
    >>>                 num_partitions=[world_size])
    >>> tensor_spec = ColoTensorSpec(pg, shard_spec)
    >>> colo_t2 = ColoTensor.from_torch_tensor(t_ref.clone(), tensor_spec)
    2. use static method from_torch_tensor
    >>> colo_t = ColoTensor.from_torch_tensor(torch.randn(2,3), spec = ColoTensorSpec(pg, ReplicaSpec())
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
            self.dist_spec = ReplicaSpec()
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
        # if the new pg is the same as the old pg, just returns
        if self.process_group == pg:
            return
        assert self.process_group.tp_world_size() == 1, \
            "Can not set_process_group on a ColoTensor whose process_group has tp world group"
        assert self.dist_spec.placement.value == 'r', \
            "Can not set_process_group on a ColoTensor whose dist spec is not REPLICATE"

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
        self._redistribute(dist_spec)

    def set_tensor_spec(self, dist_spec, compute_spec):
        if dist_spec is not None:
            assert isinstance(dist_spec, _DistSpec), f"{type(dist_spec)}"
            self.set_dist_spec(dist_spec)
        if compute_spec is not None:
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
                colo_spec = _get_spec_from_args(args, kwargs)
                return _convert_output(ret, colo_spec)

    def __repr__(self):
        return f'ColoTensor:\n{super().__repr__()}\n{self.dist_spec}\n{self.process_group}\n{self.compute_spec}'

    def _redistribute(self, dist_spec: _DistSpec) -> None:
        """_redistribute 
        Note the function will not handle the logic of backward propagation!
        It is used during model tensor initializations as an internal function.
        Args:
            dist_spec (_DistSpec): the target dist. spec.
        """
        assert self.grad_fn is None, "Current tensor has grad_fn and it can't get converted"
        with DistSpecManager.no_grad():
            self.data = DistSpecManager.handle_trans_spec(self.data, self.dist_spec, dist_spec, self.process_group)
        self.dist_spec = dist_spec

    def redistribute(self, dist_spec: _DistSpec, pg: Optional[ProcessGroup] = None) -> 'ColoTensor':
        """redistribute 
        Redistribute the tensor among processes. The rule is like this:
        1. If the pg is None, then redistributed tensor payload among TP process group. Keep the
        DP process group still as replicated.
        2. If the pg is not not None and not equal to the cureent process group.
        First, convert the tensor as replicated among TP process group.
        Second, reset the process group.
        Third, conver the tensor (new replicated both among tp and dp process group) to the new dist_spec.

        Args:
            dist_spec (_DistSpec): the new dist spec.
            pg (Optional[ProcessGroup], optional): the new process group . Defaults to None.

        Returns:
            ColoTensor: a redistributed colotensor
        """
        if pg is not None and pg != self.get_process_group():
            # if the pg is not equal, convert the current tensor to replicated
            handled = self.redistribute(ReplicaSpec())
        else:
            handled = self
            pg = self.process_group

        ret = DistSpecManager.handle_trans_spec(handled, handled.dist_spec, dist_spec, pg)
        return ColoTensor.from_torch_tensor(ret, ColoTensorSpec(pg=pg, dist_attr=dist_spec))

    def to_replicate_(self):
        """to_replicate_ 
        an inline member function, converting dist spec of the tensor to REPLICATE
        """
        self._redistribute(dist_spec=ReplicaSpec())

    def to_replicate(self) -> 'ColoTensor':
        """to_replicate
        converting dist spec of the tensor to REPLICATE
        """
        return self.redistribute(ReplicaSpec())

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

    # override builtin functions which must use tensor in replicate placement #

    def size_local(self, *args) -> torch.Size:
        with torch._C.DisableTorchFunction():
            return super().size(*args)

    def size_global(self, *args) -> torch.Size:
        """override the torch buildin size()
        the shape passed in must be in a replicate placement.
        Returns:
            ColoTensor: a tensor after viewed.
        """
        if self.is_replicate():
            return self.size_local(*args)
        spec = self.dist_spec
        dims = spec.dims
        num_partitions = spec.num_partitions
        # import inspect
        # print(*['{:40}| {}:{}\n'.format(x.function, x.filename, x.lineno) for x in inspect.stack()])
        size_list = list(self.size_local())
        for dim, num_partition in zip(dims, num_partitions):
            size_list[dim] *= num_partition
        if args == ():
            return torch.Size(size_list)
        else:
            return size_list[args[0]]

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
