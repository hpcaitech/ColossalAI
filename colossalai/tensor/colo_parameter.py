from .colo_tensor import ColoTensor
from .const import TensorType
import torch
from colossalai.tensor import TensorSpec, distspec
from copy import copy
from .param_op_hook import _ParamOpHookWrapper, PreFwdPostBwd, PostFwdPreBwd


class ColoParameter(ColoTensor):
    r"""A kind of ColoTensor to be considered as a module parameter.

    """

    def __new__(cls,
                data: torch.Tensor,
                requires_grad: bool = True,
                spec: TensorSpec = TensorSpec(distspec.replicate())) -> 'ColoParameter':
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(self,
                 data: torch.Tensor,
                 requires_grad: bool = True,
                 spec: TensorSpec = TensorSpec(distspec.replicate())) -> None:
        self._spec = copy(spec)
        self._type = TensorType.MODEL
        self._graph_node = None

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
                if len(params) > 0:
                    with torch._C.DisableTorchFunction():
                        args = PreFwdPostBwd.apply(params, *args)
                    ret = super().__torch_function__(func, types, args, kwargs)
                    with torch._C.DisableTorchFunction():
                        ret = PostFwdPreBwd.apply(params, ret)
                    return ret
        return super().__torch_function__(func, types, args, kwargs)
