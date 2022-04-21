import torch
from .api import _STATEFUL_OPS


class StatefulTensorV2(object):

    def __new__(cls, *args, **kwargs):
        return super(StatefulTensorV2, cls).__new__(cls)

    def __init__(self, t: torch.Tensor) -> None:
        self._torch_tensor = t

    def torch_tensor(self) -> torch.Tensor:
        return self._torch_tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        global _STATEFUL_OPS
        if func in _STATEFUL_OPS:
            # Find StatefulTensorV2 instance to get process_group.
            for arg in args:
                if isinstance(arg, StatefulTensorV2):
                    return _STATEFUL_OPS[func](types, args, kwargs, None)

            for kwarg in kwargs.values():
                if isinstance(kwarg, StatefulTensorV2):
                    return _STATEFUL_OPS[func](types, args, kwargs, None)

        raise RuntimeError(f"torch function '{func.__name__}', with args: {args} and "
                           f"kwargs: {kwargs} not supported for StatefulTensorV2!")
