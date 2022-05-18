from .op_wrapper import _COLOSSAL_OPS
from copy import copy
import torch
from typing import Tuple, Optional, Callable, Union
from numpy import product
from colossalai.tensor import TensorSpec
from .const import TensorType
from colossalai.tensor import dist_spec
from colossalai.tensor.dist_spec_mgr import DistSpecManager
from colossalai.tensor.dist_spec import _DistSpec


class ColoTensor(object):
    """ Data Structure for Tensor in Colossal-AI
    1. It contains a torch.Tensor as an attribute.
    2. It supports lazy init the tensor's payload.
    3. It can hijack the torch functions which using ColoTensors as args to our customized functions.
    4. It supports distributing the tensor's payload to the shards among processes. (TODO)
    """

    def __new__(cls, *args, **kwargs):
        return super(ColoTensor, cls).__new__(cls)

    def __init__(self,
                 *size: Tuple[int],
                 dtype=None,
                 requires_grad=False,
                 pin_memory=False,
                 device=None,
                 torch_tensor=torch.empty(0),
                 spec: TensorSpec = TensorSpec(dist_spec.replicate())):
        self._size = size
        self._dtype = dtype
        self._requires_grad = requires_grad
        self._pin_memory = pin_memory
        self._device = device
        self._torch_tensor = torch_tensor
        self._spec = copy(spec)
        self._type = TensorType.NONMODEL
        self._graph_node = None

    def __getitem__(self, key):
        return ColoTensor.init_from_torch_tensor(self.torch_tensor()[key])

    @property
    def spec(self) -> TensorSpec:
        return self._spec

    @property
    def shard_pattern(self):
        return self._shard_pattern

    @property
    def data(self):
        return self._torch_tensor.data

    @data.setter
    def data(self, tensor: Union[torch.Tensor, "ColoTensor"]):
        if isinstance(tensor, ColoTensor):
            self._torch_tensor.data = tensor.data
        elif isinstance(tensor, torch.Tensor):
            self._torch_tensor.data = tensor
        else:
            raise NotImplementedError

    @property
    def grad(self):
        return self._torch_tensor.grad

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return torch.Size(self._size)

    @property
    def device(self):
        return self._torch_tensor.device

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self._size[dim]

    def dim(self):
        return len(self._size)

    def normal_(self, mean=0., std=1.):
        torch_tensor = self.torch_tensor()
        return torch_tensor.normal_(mean=mean, std=std)

    def numel(self):
        return product(self._size)

    @staticmethod
    def init_from_torch_tensor(tensor: torch.Tensor,
                               save_payload=True,
                               spec: TensorSpec = TensorSpec(dist_spec.replicate())) -> 'ColoTensor':
        colo_t = ColoTensor(*tensor.size(),
                            dtype=tensor.dtype,
                            requires_grad=tensor.requires_grad,
                            pin_memory=tensor.is_pinned(),
                            device=tensor.device,
                            torch_tensor=tensor if save_payload else torch.empty(0),
                            spec=spec)
        return colo_t

    def del_torch_tensor(self, save_shape=False) -> None:
        """
        delete the payload of the torch tensor.

        Args:
            save_shape (bool, optional): if saving the shape of the torch_tensor. 
            If saving the shape, the size of self._torch_tensor is inconsist with the self._size.
            Defaults to False.
        """
        if not save_shape:
            self._size = (0,)
        self._torch_tensor = torch.empty((0,), device=self._device, dtype=self._dtype)

    def torch_tensor(self) -> torch.Tensor:
        if self._torch_tensor.numel() == 0:
            self._torch_tensor = torch.empty(*self._size,
                                             dtype=self._dtype,
                                             pin_memory=self._pin_memory,
                                             requires_grad=self._requires_grad,
                                             device=self._device)
        return self._torch_tensor

    def set_spec(self, spec: TensorSpec) -> None:
        spec = copy(spec)
        self.to_dist_spec(spec.dist_spec)
        self._spec = spec

    def has_spec(self) -> bool:
        return self._spec.num_action > 0

    def is_model_data(self) -> bool:
        return self._type == TensorType.MODEL

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        global _COLOSSAL_OPS
        if func in _COLOSSAL_OPS:
            for arg in args:
                if isinstance(arg, ColoTensor):
                    return _COLOSSAL_OPS[func](types, args, kwargs, None)

            for kwarg in kwargs.values():
                if isinstance(kwarg, ColoTensor):
                    return _COLOSSAL_OPS[func](types, args, kwargs, None)
        else:
            # If we have not hijact the function, convert the ColoTensors in args and kwargs to torch tensors.
            args = [arg.torch_tensor() if isinstance(arg, ColoTensor) else arg for arg in args]
            if kwargs is None:
                kwargs = {}

            kwargs = {k: v.torch_tensor() if isinstance(v, ColoTensor) else v for k, v in kwargs.items()}
            return cls._filter_outputs_with_colo(func(*args, **kwargs))

    def backward(self, gradient: Optional[torch.Tensor] = None, retain_graph: bool = False):
        self._torch_tensor.backward(gradient=gradient, retain_graph=retain_graph)

    def __add__(self, o) -> "ColoTensor":
        if isinstance(o, ColoTensor):
            return ColoTensor.init_from_torch_tensor(self.torch_tensor() + o.torch_tensor())
        elif isinstance(o, (torch.Tensor, int, float)):
            return ColoTensor.init_from_torch_tensor(self.torch_tensor() + o)
        else:
            raise TypeError(f'{type(o)} is not supported in ColoTensor __add__')

    __radd__ = __add__

    def __truediv__(self, o) -> "ColoTensor":
        return ColoTensor.init_from_torch_tensor(self.torch_tensor() / o)

    def __getattr__(self, name):

        def replace_tensor_with_colo(func):

            def execute_func(*args, **kwargs):
                # transform the ColoTensor args to torch Tensor.
                args = [arg.torch_tensor() if isinstance(arg, ColoTensor) else arg for arg in args]
                if kwargs is None:
                    kwargs = {}
                kwargs = {k: v.torch_tensor() if isinstance(v, ColoTensor) else v for k, v in kwargs.items()}
                return self._filter_outputs_with_colo(func(*args, **kwargs))

            return execute_func

        if hasattr(self._torch_tensor, name) == False:
            raise AttributeError

        attr = getattr(self._torch_tensor, name)

        if isinstance(attr, Callable):
            return replace_tensor_with_colo(attr)
        else:
            return attr

    @classmethod
    def _filter_outputs_with_colo(cls, outputs):
        if outputs is None:    # return None
            return None
        elif type(outputs) is not tuple:    # num of return val = 1
            return ColoTensor.init_from_torch_tensor(outputs) if type(outputs) is torch.Tensor else outputs
        else:    # num of return val > 1
            return tuple([
                ColoTensor.init_from_torch_tensor(output) if type(output) is torch.Tensor else output
                for output in outputs
            ])

    def __mul__(self, other) -> "ColoTensor":
        if isinstance(other, ColoTensor):
            return ColoTensor.init_from_torch_tensor(self.torch_tensor() * other.torch_tensor())
        elif isinstance(other, (torch.Tensor, int, float)):
            return ColoTensor.init_from_torch_tensor(self.torch_tensor() * other)
        else:
            raise TypeError(f'{type(other)} is not supported in ColoTensor __mul__')

    __rmul__ = __mul__

    def to_dist_spec(self, dist_spec: _DistSpec) -> None:
        self._torch_tensor = DistSpecManager.handle_trans_spec(self.torch_tensor(), self.spec.dist_spec, dist_spec)
        if self._torch_tensor.is_leaf:
            self._torch_tensor.requires_grad = self._requires_grad
        self._size = self._torch_tensor.size()
        self._spec.dist_spec = dist_spec
