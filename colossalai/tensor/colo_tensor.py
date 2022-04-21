import torch
from .op_wrapper import _COLOSSAL_OPS
from typing import Tuple


class ColoTensor(object):
    """ Data Structure for Tensor in Colossal-AI
    1. It contains a torch.Tensor as an attribute.
    2. It supports lazy init the tensor's payload.
    3. It can hijack the torch functions which using ColoTensors as args to our customized functions.
    4. It supports distributing the tensor's payload to the shards among processes. (TODO)
    """

    def __new__(cls, *args, **kwargs):
        return super(ColoTensor, cls).__new__(cls)

    def __init__(
        self,
        *size: Tuple[int],
        dtype=None,
        requires_grad=False,
        pin_memory=False,
        torch_tensor=None,
    ):
        self._size = size
        self._dtype = dtype
        self._requires_grad = requires_grad
        self._pin_memory = pin_memory
        self._torch_tensor = torch_tensor

    @staticmethod
    def init_from_torch_tensor(tensor: torch.Tensor):
        colo_t = ColoTensor(*tensor.size(),
                            dtype=tensor.dtype,
                            requires_grad=tensor.requires_grad,
                            pin_memory=tensor.pin_memory,
                            torch_tensor=tensor)
        return colo_t

    def torch_tensor(self) -> torch.Tensor:
        if self._torch_tensor == None:
            self._torch_tensor = torch.empty(*self._size,
                                             dtype=self._dtype,
                                             requires_grad=self._requires_grad,
                                             pin_memory=self._pin_memory)
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
