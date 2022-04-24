from .op_wrapper import _COLOSSAL_OPS

import torch
from typing import Tuple, Optional
from numpy import product


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
            device=None,
            torch_tensor=torch.empty(0),
            shard_spec: str = None,
    ):
        self._size = size
        self._dtype = dtype
        self._requires_grad = requires_grad
        self._pin_memory = pin_memory
        self._device = device
        self._torch_tensor = torch_tensor
        self._shard_spec = shard_spec

    @property
    def shard_spec(self) -> Optional[str]:
        return self._shard_spec

    @property
    def data(self):
        return self._torch_tensor.data

    @property
    def grad(self):
        return self._torch_tensor.grad

    @property
    def size(self):
        return self._size

    def numel(self):
        return product(self._size)

    @staticmethod
    def init_from_torch_tensor(tensor: torch.Tensor, save_payload=True) -> 'ColoTensor':
        colo_t = ColoTensor(*tensor.size(),
                            dtype=tensor.dtype,
                            requires_grad=tensor.requires_grad,
                            pin_memory=tensor.is_pinned(),
                            device=tensor.device,
                            torch_tensor=tensor if save_payload else torch.empty(0))
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
            return func(*args, **kwargs)
