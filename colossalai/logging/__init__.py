from typing import List
from .logging import DistributedLogger
import logging

__all__ = ['get_dist_logger', 'DistributedLogger']


def get_dist_logger(name='colossalai'):
    """Get logger instance based on name. The DistributedLogger will create singleton instances,
    which means that only one logger instance is created per name.

    :param name: name of the logger, name must be unique
    :type name: str

    :return: a distributed logger instance
    :rtype: :class:`colossalai.logging.DistributedLogger`
    """
    return DistributedLogger.get_instance(name=name)


def disable_existing_loggers(except_loggers: List[str] = ['colossalai']):
    """Set the level of existing loggers to `WARNING`.

    :param except_loggers: loggers in this `list` will be ignored when disabling, defaults to ['colossalai']
    :type except_loggers: list, optional
    """
    for log_name in logging.Logger.manager.loggerDict.keys():
        if log_name not in except_loggers:
            logging.getLogger(log_name).setLevel(logging.WARNING)
