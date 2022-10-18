import functools

from colossalai.logging import get_dist_logger
from colossalai.tensor.sharding_spec import ShardingSpecException

__all__ = ['ignore_sharding_exception']


def ignore_sharding_exception(func):
    """
    A function wrapper to handle the ShardingSpecException in the function.
    If ShardingSpecException occurs, this function will return None.

    Usage:
        # mute the assertion error in the function
        @ignore_sharding_exception
        def do_something():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger = get_dist_logger()
            rst = func(*args, **kwargs)
            return rst
        except ShardingSpecException as e:
            logger.debug(e)
            return None

    return wrapper
