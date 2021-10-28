#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging
from pathlib import Path

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc

_FORMAT = 'colossalai - %(name)s - %(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=_FORMAT)


class DistributedLogger:
    """This is a distributed event logger class essentially based on :class:`logging`.

    :param name: The name of the logger
    :type name: str
    :param level: The threshold for the logger. Logging messages which are less severe than `level`
        will be ignored
    :type level: str
    :param root_path: The root path where logs are stored
    :type root_path: str, optional
    :param mode: The mode that the file is opened in. Defaults to 'a'
    :type mode: str, optional
    """

    def __init__(self, name, level='INFO', root_path: str = None, mode='a'):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level))

        if root_path is not None:
            log_root_path = Path(root_path)
            # create path if not exists
            log_root_path.mkdir(parents=True, exist_ok=True)
            log_path = log_root_path.joinpath(f'{name}.log')
            file_handler = logging.FileHandler(log_path, mode)
            file_handler.setLevel(getattr(logging, level))
            formatter = logging.Formatter(_FORMAT)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def _log(self, level, message: str, parallel_mode: ParallelMode = ParallelMode.GLOBAL, ranks: list = None):
        if ranks is None:
            getattr(self._logger, level)(message)
        else:
            local_rank = gpc.get_local_rank(parallel_mode)
            if local_rank in ranks:
                getattr(self._logger, level)(message)

    def info(self, message: str, parallel_mode: ParallelMode = ParallelMode.GLOBAL, ranks: list = None):
        """Stores an info log message.

        :param message:
        :type message:
        :param parallel_mode:
        :type parallel_mode:
        :param ranks:
        :type ranks:
        """
        self._log('info', message, parallel_mode, ranks)

    def warning(self, message: str, parallel_mode: ParallelMode = ParallelMode.GLOBAL, ranks: list = None):
        """Stores a warning log message.

        :param message: The message to be logged
        :type message: str
        :param parallel_mode: The parallel mode used for logging. Defaults to ParallelMode.GLOBAL
        :type parallel_mode: :class:`colossalai.context.parallel_mode.ParallelMode`
        :param ranks: List of parallel ranks
        :type ranks: list
        """
        self._log('warning', message, parallel_mode, ranks)

    def debug(self, message: str, parallel_mode: ParallelMode = ParallelMode.GLOBAL, ranks: list = None):
        """Stores a debug log message.

        :param message: The message to be logged
        :type message: str
        :param parallel_mode: The parallel mode used for logging. Defaults to ParallelMode.GLOBAL
        :type parallel_mode: :class:`colossalai.context.parallel_mode.ParallelMode`
        :param ranks: List of parallel ranks
        :type ranks: list
        """
        self._log('debug', message, parallel_mode, ranks)

    def error(self, message: str, parallel_mode: ParallelMode = ParallelMode.GLOBAL, ranks: list = None):
        """Stores an error log message.

        :param message: The message to be logged
        :type message: str
        :param parallel_mode: The parallel mode used for logging. Defaults to ParallelMode.GLOBAL
        :type parallel_mode: :class:`colossalai.context.parallel_mode.ParallelMode`
        :param ranks: List of parallel ranks
        :type ranks: list
        """
        self._log('error', message, parallel_mode, ranks)
