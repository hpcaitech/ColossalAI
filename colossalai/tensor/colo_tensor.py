from functools import lru_cache
from typing import Callable, Set

import torch

INPALCE_MAPPING = {
    torch.Tensor.add_: torch.Tensor.add,
    torch.Tensor.sub_: torch.Tensor.sub,
    torch.Tensor.mul_: torch.Tensor.mul,
    torch.Tensor.div_: torch.Tensor.div,
}


@lru_cache(None)
def _get_my_nowrap_functions() -> Set[Callable]:
    Tensor = torch.Tensor
    return {
        Tensor._base.__get__,
        Tensor.grad.__get__,
        Tensor._grad.__get__,
        Tensor.data.__get__,  # make .data returns torch.Tensor rather than ColoTensor
    }


def _convert(output):
    if isinstance(output, torch.Tensor) and not isinstance(output, ColoTensor):
        output.__class__ = ColoTensor
    elif isinstance(output, (list, tuple)):
        output = type(output)(_convert(o) for o in output)
    return output


def _convert_output(output, func):
    if func in _get_my_nowrap_functions():
        return output
    return _convert(output)


class ColoTensor(torch.Tensor):
    """Data Structure for Tensor in Colossal-AI. It is a subclass of torch.Tensor.

    It is only used to trigger the torch function hook.

    Args:
        data (torch.Tensor): a torch tensor used as the payload the colotensor.
    """

    torch_major = int(torch.__version__.split(".")[0])
    torch_minor = int(torch.__version__.split(".")[1])

    def __new__(cls, data: torch.Tensor) -> "ColoTensor":
        """
        The signature of the __new__ has to be consistent with the torch.Tensor.

        Args:
            data (torch.Tensor): a torch tensor used as the payload the colotensor.

        Returns:
            ColoTensor: a ColoTensor wrappers the data.
        """
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, data.requires_grad)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        if cls.torch_major > 1 or (cls.torch_major == 1 and cls.torch_minor >= 12):
            # in order to trigger pre-op hook in the forward of checkpoint module
            # we have to capture the `backward` function
            # and make sure that it does not in `torch._C.DisableTorchFunction()` context
            if func is torch.Tensor.backward:
                assert len(args) == 1  # only has 1 parameter
                backward_tensor = torch.Tensor(args[0])
                tensor_kwargs = {k: torch.Tensor(v) if torch.is_tensor(v) else v for k, v in kwargs.items()}
                return backward_tensor.backward(**tensor_kwargs)

        # replace the in-place function
        if func in INPALCE_MAPPING:
            func = INPALCE_MAPPING[func]
        # set the 'inplace' kwargs to False
        if "inplace" in kwargs:
            kwargs["inplace"] = False

        with torch._C.DisableTorchFunction():
            ret = func(*args, **kwargs)
            return _convert_output(ret, func)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            with torch._C.DisableTorchFunction():
                data = self.data.clone()
            tensor = ColoTensor(data)
            memo[id(self)] = tensor
            return tensor
