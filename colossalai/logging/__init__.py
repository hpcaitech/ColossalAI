from .logging import DistributedLogger

__all__ = ['get_dist_logger', 'DistributedLogger']


def get_dist_logger(name='root'):
    """Get logger instance based on name. The DistributedLogger will create singleton instances,
    which means that only one logger instance is created per name.

    :param name: name of the logger, name must be unique
    :type name: str

    :return: a distributed logger instance
    :rtype: :class:`colossalai.logging.DistributedLogger`
    """
    return DistributedLogger.get_instance(name=name)
