#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import colossalai
import logging
from pathlib import Path
from typing import Union

from colossalai.context.parallel_mode import ParallelMode


_FORMAT = 'colossalai - %(name)s - %(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=_FORMAT)


class DistributedLogger:
    """This is a distributed event logger class essentially based on :class:`logging`.

    :param name: The name of the logger
    :type name: str
    """

    __instances = dict()

    @staticmethod
    def get_instance(name: str):
        """Get the unique single logger instance based on name.

        :param name: The name of the logger
        :type name: str
        :return: A DistributedLogger object
        :rtype: DistributedLogger
        """
        if name in DistributedLogger.__instances:
            return DistributedLogger.__instances[name]
        else:
            logger = DistributedLogger(name=name)
            return logger

    def __init__(self, name):
        if name in DistributedLogger.__instances:
            raise Exception('Logger with the same name has been created, you should use colossalai.logging.get_dist_logger')
        else:
            self._name = name
            self._logger = logging.getLogger(name)
            DistributedLogger.__instances[name] = self

    @staticmethod
    def _check_valid_logging_level(level: str):
        assert level in ['INFO', 'DEBUG', 'WARNING', 'ERROR'], 'found invalid logging level'

    def set_level(self, level: str):
        """Set the logging level

        :param level: Can only be INFO, DEBUG, WARNING and ERROR
        :type level: str
        """
        self._check_valid_logging_level(level)
        self._logger.setLevel(getattr(logging, level))

    def log_to_file(self,
                    path: Union[str, Path],
                    mode: str = 'a',
                    level: str = 'INFO',
                    suffix: str = None):
        """Save the logs to file

        :param path: The file to save the log
        :type path: A string or pathlib.Path object
        :param mode: The mode to write log into the file
        :type mode: str
        :param level: Can only be INFO, DEBUG, WARNING and ERROR
        :type level: str
        :param suffix: The suffix string of log's name
        :type suffix: str
        """
        assert isinstance(path, (str, Path)), \
            f'expected argument path to be type str or Path, but got {type(path)}'
        self._check_valid_logging_level(level)
        if isinstance(path, str):
            path = Path(path)

        # set the default file name if path is a directory
        if not colossalai.core.global_context.is_initialized(ParallelMode.GLOBAL):
            rank = 0
        else:
            rank = colossalai.core.global_context.get_global_rank()

        if suffix is not None:
            log_file_name = f'rank_{rank}_{suffix}.log'
        else:
            log_file_name = f'rank_{rank}.log'
        path = path.joinpath(log_file_name)

        # add file handler
        file_handler = logging.FileHandler(path, mode)
        file_handler.setLevel(getattr(logging, level))
        formatter = logging.Formatter(_FORMAT)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

    def _log(self, level, message: str, parallel_mode: ParallelMode = ParallelMode.GLOBAL, ranks: list = None):
        if ranks is None:
            getattr(self._logger, level)(message)
        else:
            local_rank = colossalai.core.global_context.get_local_rank(parallel_mode)
            if local_rank in ranks:
                getattr(self._logger, level)(message)

    def info(self, message: str, parallel_mode: ParallelMode = ParallelMode.GLOBAL, ranks: list = None):
        """Log an info message.

        :param message: The message to be logged
        :type message: str
        :param parallel_mode: The parallel mode used for logging. Defaults to ParallelMode.GLOBAL
        :type parallel_mode: :class:`colossalai.context.parallel_mode.ParallelMode`
        :param ranks: List of parallel ranks
        :type ranks: list
        """
        self._log('info', message, parallel_mode, ranks)

    def warning(self, message: str, parallel_mode: ParallelMode = ParallelMode.GLOBAL, ranks: list = None):
        """Log a warning message.

        :param message: The message to be logged
        :type message: str
        :param parallel_mode: The parallel mode used for logging. Defaults to ParallelMode.GLOBAL
        :type parallel_mode: :class:`colossalai.context.parallel_mode.ParallelMode`
        :param ranks: List of parallel ranks
        :type ranks: list
        """
        self._log('warning', message, parallel_mode, ranks)

    def debug(self, message: str, parallel_mode: ParallelMode = ParallelMode.GLOBAL, ranks: list = None):
        """Log a debug message.

        :param message: The message to be logged
        :type message: str
        :param parallel_mode: The parallel mode used for logging. Defaults to ParallelMode.GLOBAL
        :type parallel_mode: :class:`colossalai.context.parallel_mode.ParallelMode`
        :param ranks: List of parallel ranks
        :type ranks: list
        """
        self._log('debug', message, parallel_mode, ranks)

    def error(self, message: str, parallel_mode: ParallelMode = ParallelMode.GLOBAL, ranks: list = None):
        """Log an error message.

        :param message: The message to be logged
        :type message: str
        :param parallel_mode: The parallel mode used for logging. Defaults to ParallelMode.GLOBAL
        :type parallel_mode: :class:`colossalai.context.parallel_mode.ParallelMode`
        :param ranks: List of parallel ranks
        :type ranks: list
        """
        self._log('error', message, parallel_mode, ranks)
