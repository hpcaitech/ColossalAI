import functools
import warnings

__all__ = ['exception_handler']


def exception_handler(func):
    """
    A function wrapper to handle the AssertionError in the function.

    Usage:
        # mute the assertion error in the function
        @exception_handler
        def do_something():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            rst = func(*args, **kwargs)
            return rst
        except AssertionError as e:
            warnings.warn(f'{e}')

    return wrapper
