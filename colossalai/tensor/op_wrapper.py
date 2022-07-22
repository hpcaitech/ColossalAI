from typing import (
    Callable,
    Dict,
)
import functools

# Custom sharded ops
_COLOSSAL_OPS: Dict[str, Callable] = {}


def _register_colo_op(op, func):
    global _COLOSSAL_OPS
    _COLOSSAL_OPS[op] = func


def colo_op_impl(func):
    """
    Provides a way for users to write their own custom operator. This
    can be used to override existing ColoTensor operators or write a new
    one not supported by ColoTensor. If the operator in question is covered
    by ``__torch_function__`` dispatch and has a ColoTensor as any of its
    parameters, the function provided will be invoked for that operator.

    Example:
        >>> @colo_op_impl(torch.nn.functional.linear)
        >>> def my_custom_linear(types, args, kwargs, process_group):
        >>>   ....
        >>>
        >>> input = torch.rand(10, 32)
        >>> weight = ColoTensor(torch.rand(32, 16))
        >>> bias = ColoTensor(torch.rand(16))
        >>> # This will call `my_custom_linear` instead of the default.
        >>> torch.nn.functional.linear(input, weight, bias)

    The types, args and kwargs parameters are the same parameters that are
    passed to ``__torch_function__`` dispatch API
    (https://pytorch.org/docs/stable/notes/extending.html#extending-torch).

    Args:
        func(Callable): Torch function for which we want to provide a sharded
            implementation (ex: torch.nn.functional.linear)
    """

    def decorator_sharded_func(wrapped_func):
        _register_colo_op(func, wrapped_func)

        @functools.wraps(wrapped_func)
        def wrapper(*args, **kwargs):
            return wrapped_func(*args, **kwargs)

        return wrapper

    return decorator_sharded_func
