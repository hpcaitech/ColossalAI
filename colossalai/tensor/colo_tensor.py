import torch
from .op_wrapper import _COLOSSAL_OPS


class ColoTensor(object):

    def __new__(cls, *args, **kwargs):
        return super(ColoTensor, cls).__new__(cls)

    def __init__(self, t: torch.Tensor) -> None:
        self._torch_tensor = t

    def torch_tensor(self) -> torch.Tensor:
        return self._torch_tensor

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

            kwargs = {
                kwarg: kwargs[kwarg].torch_tensor() if isinstance(kwarg, ColoTensor) else kwarg for kwarg in kwargs
            }
            return func(*args, **kwargs)
