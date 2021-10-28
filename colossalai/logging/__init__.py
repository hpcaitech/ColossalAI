from colossalai.core import global_context as gpc
from .logging import DistributedLogger

__all__ = ['get_global_dist_logger', 'get_dist_logger', 'DistributedLogger', 'init_global_dist_logger']

_GLOBAL_LOGGER: DistributedLogger = None


def get_dist_logger(name, level='INFO', root_path: str = None, mode='a'):
    return DistributedLogger(name=name, level=level, root_path=root_path, mode=mode)


def get_global_dist_logger():
    assert _GLOBAL_LOGGER is not None, 'Global distributed logger is not initialized'
    return _GLOBAL_LOGGER


def init_global_dist_logger():
    rank = gpc.get_global_rank()
    if hasattr(gpc.config, 'logging'):
        logger = get_dist_logger(name=f'rank_{rank}', **gpc.config.logging)
    else:
        logger = get_dist_logger(name=f'rank_{rank}', level='INFO')
    global _GLOBAL_LOGGER
    assert _GLOBAL_LOGGER is None, 'Global distributed logger has already been initialized'
    _GLOBAL_LOGGER = logger
