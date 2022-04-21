from typing import (
    Callable,
    Dict,
)

# Custom sharded ops
_STATEFUL_OPS: Dict[str, Callable] = {}


def _register_stateful_op(op, func):
    from inspect import signature
    if len(signature(func).parameters) != 4:
        raise TypeError(f'Custom stateful op function expects signature: '
                        f'(types, args, kwargs, process_group), but received '
                        f'signature: {signature(func)}')
    global _STATEFUL_OPS
    _STATEFUL_OPS[op] = func
