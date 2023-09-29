from typing import Callable

import torch

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 12:
    META_COMPATIBILITY = False
elif TORCH_MAJOR == 1 and TORCH_MINOR == 12:
    META_COMPATIBILITY = True
elif TORCH_MAJOR == 1 and TORCH_MINOR == 13:
    META_COMPATIBILITY = True
elif TORCH_MAJOR == 2:
    META_COMPATIBILITY = True


def compatibility(is_backward_compatible: bool = False) -> Callable:
    """A decorator to make a function compatible with different versions of PyTorch.

    Args:
        is_backward_compatible (bool, optional): Whether the function is backward compatible. Defaults to False.

    Returns:
        Callable: The decorated function
    """

    def decorator(func):
        if META_COMPATIBILITY:
            return func
        else:
            if is_backward_compatible:
                return func
            else:

                def wrapper(*args, **kwargs):
                    raise RuntimeError(f"Function `{func.__name__}` is not compatible with PyTorch {torch.__version__}")

                return wrapper

    return decorator


def is_compatible_with_meta() -> bool:
    """Check the meta compatibility. Normally it should be called before importing some of the `colossalai.fx`
    modules. If the meta compatibility is not satisfied, the `colossalai.fx` modules will be replaced by its
    experimental counterparts.

    Returns:
        bool: The meta compatibility
    """
    return META_COMPATIBILITY
