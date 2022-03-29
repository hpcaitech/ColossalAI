import logging
from typing import List, Optional

from .logger import DistributedLogger

__all__ = ['get_dist_logger', 'DistributedLogger', 'disable_existing_loggers']


def get_dist_logger(name='colossalai'):
    """Get logger instance based on name. The DistributedLogger will create singleton instances,
    which means that only one logger instance is created per name.

    :param name: name of the logger, name must be unique
    :type name: str

    :return: a distributed logger instance
    :rtype: :class:`colossalai.logging.DistributedLogger`
    """
    return DistributedLogger.get_instance(name=name)


def disable_existing_loggers(include: Optional[List[str]] = None, exclude: List[str] = ['colossalai']):
    """Set the level of existing loggers to `WARNING`. By default, it will "disable" all existing loggers except the logger named "colossalai".

    Args:
        include (Optional[List[str]], optional): Loggers whose name in this list will be disabled.
            If set to `None`, `exclude` argument will be used. Defaults to None.
        exclude (List[str], optional): Loggers whose name not in this list will be disabled.
            This argument will be used only when `include` is None. Defaults to ['colossalai'].
    """
    if include is None:
        filter_func = lambda name: name not in exclude
    else:
        filter_func = lambda name: name in include

    for log_name in logging.Logger.manager.loggerDict.keys():
        if filter_func(log_name):
            logging.getLogger(log_name).setLevel(logging.WARNING)
