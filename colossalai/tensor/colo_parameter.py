from .colo_tensor import ColoTensor
from .const import TensorType
import torch
from colossalai.tensor import TensorSpec, distspec
from copy import copy
from .param_op_hook import _ParamOpHookWrapper, PreFwdPostBwd, PostFwdPreBwd
from typing import Optional


class ColoParameter(ColoTensor, torch.nn.Parameter):
    r"""A kind of ColoTensor to be considered as a module parameter.

    """

    def __new__(cls,
                data: Optional[torch.Tensor] = None,
                requires_grad: bool = True,
                spec: TensorSpec = TensorSpec(distspec.replicate())) -> 'ColoParameter':
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(self,
                 data: Optional[torch.Tensor] = None,
                 requires_grad: bool = True,
                 spec: TensorSpec = TensorSpec(distspec.replicate())) -> None:
        self._spec = copy(spec)
        self._type = TensorType.MODEL
        self._graph_node = None

        # a list contains modules sharing this ColoParameter with others.
        self._shared_param_modules = []

    @property
    def shared_param_modules(self):
        return self._shared_param_modules

    @staticmethod
    def from_torch_tensor(tensor: torch.Tensor,
                          requires_grad: bool = True,
                          spec: TensorSpec = TensorSpec(distspec.replicate())) -> 'ColoParameter':
        tensor = tensor.as_subclass(ColoParameter)
        tensor.__init__(tensor, requires_grad=requires_grad, spec=spec)
        return tensor

    def __repr__(self):
        return f'ColoParameter: {torch.Tensor.__repr__(self)}'

    @classmethod
    def __torch_function__(cls, func, types, args=..., kwargs=None):
        if len(_ParamOpHookWrapper.hooks) > 0:
            if not func.__name__.startswith('__'):
                params = list(filter(lambda arg: isinstance(arg, ColoParameter), args))
                if kwargs is not None:
                    params.extend(list(filter(lambda arg: isinstance(arg, ColoParameter), kwargs.values())))
                if len(params) > 0:
                    with torch._C.DisableTorchFunction():
                        args = PreFwdPostBwd.apply(params, *args)
                    ret = super().__torch_function__(func, types, args, kwargs)
                    with torch._C.DisableTorchFunction():
                        ret = PostFwdPreBwd.apply(params, ret)
                    return ret
        return super().__torch_function__(func, types, args, kwargs)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            with torch._C.DisableTorchFunction():
                data = self.data.clone()
            tensor = ColoParameter(data, self.requires_grad, spec=copy(self.spec))
            memo[id(self)] = tensor
            return tensor

    def __reduce_ex__(self, proto):
        # Adapted from torch._utils._rebuild_parameter
        # def _rebuild_colo_parameter(data, requires_grad, backward_hooks):
        #     colo_param = ColoParameter(data, requires_grad)
        #     colo_param._backward_hooks = backward_hooks
        #     return colo_param

        # return (
        #     _rebuild_colo_parameter,
        #     (self.data, self.requires_grad, OrderedDict())
        # )

        # TODO(jzy) we don't support object reflection now.
        # distspec cannot be pickled or rebuilt because it's tightly connected to runtime attribute `process_group`.
        raise NotImplementedError
