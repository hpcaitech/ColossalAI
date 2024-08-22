#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import inspect
import logging
from pathlib import Path
from typing import List, Union

import torch.distributed as dist


class DistributedLogger:
    """This is a distributed event logger class essentially based on :class:`logging`.

    Args:
        name (str): The name of the logger.

    Note:
        The parallel_mode used in ``info``, ``warning``, ``debug`` and ``error``
        should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """

    __instances = dict()

    @staticmethod
    def get_instance(name: str):
        """Get the unique single logger instance based on name.

        Args:
            name (str): The name of the logger.

        Returns:
            DistributedLogger: A DistributedLogger object
        """
        if name in DistributedLogger.__instances:
            return DistributedLogger.__instances[name]
        else:
            logger = DistributedLogger(name=name)
            return logger

    def __init__(self, name):
        if name in DistributedLogger.__instances:
            raise Exception(
                "Logger with the same name has been created, you should use colossalai.logging.get_dist_logger"
            )
        else:
            handler = None
            formatter = logging.Formatter("colossalai - %(name)s - %(levelname)s: %(message)s")
            try:
                from rich.logging import RichHandler

                handler = RichHandler(show_path=False, markup=True, rich_tracebacks=True)
                handler.setFormatter(formatter)
            except ImportError:
                handler = logging.StreamHandler()
                handler.setFormatter(formatter)

            self._name = name
            self._logger = logging.getLogger(name)
            self._logger.setLevel(logging.INFO)
            if handler is not None:
                self._logger.addHandler(handler)
            self._logger.propagate = False

            DistributedLogger.__instances[name] = self

    @property
    def rank(self):
        return dist.get_rank() if dist.is_initialized() else 0

    @staticmethod
    def __get_call_info():
        stack = inspect.stack()

        # stack[1] gives previous function ('info' in our case)
        # stack[2] gives before previous function and so on

        fn = stack[2][1]
        ln = stack[2][2]
        func = stack[2][3]

        return fn, ln, func

    @staticmethod
    def _check_valid_logging_level(level: str):
        assert level in ["INFO", "DEBUG", "WARNING", "ERROR"], "found invalid logging level"

    def set_level(self, level: str) -> None:
        """Set the logging level

        Args:
            level (str): Can only be INFO, DEBUG, WARNING and ERROR.
        """
        self._check_valid_logging_level(level)
        self._logger.setLevel(getattr(logging, level))

    def log_to_file(self, path: Union[str, Path], mode: str = "a", level: str = "INFO", suffix: str = None) -> None:
        """Save the logs to file

        Args:
            path (A string or pathlib.Path object): The file to save the log.
            mode (str): The mode to write log into the file.
            level (str): Can only be INFO, DEBUG, WARNING and ERROR.
            suffix (str): The suffix string of log's name.
        """
        assert isinstance(path, (str, Path)), f"expected argument path to be type str or Path, but got {type(path)}"
        self._check_valid_logging_level(level)

        if isinstance(path, str):
            path = Path(path)

        # create log directory
        path.mkdir(parents=True, exist_ok=True)

        if suffix is not None:
            log_file_name = f"rank_{self.rank}_{suffix}.log"
        else:
            log_file_name = f"rank_{self.rank}.log"
        path = path.joinpath(log_file_name)

        # add file handler
        file_handler = logging.FileHandler(path, mode)
        file_handler.setLevel(getattr(logging, level))
        formatter = logging.Formatter("colossalai - %(name)s - %(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

    def _log(self, level, message: str, ranks: List[int] = None) -> None:
        if ranks is None:
            getattr(self._logger, level)(message)
        else:
            if self.rank in ranks:
                getattr(self._logger, level)(message)

    def info(self, message: str, ranks: List[int] = None) -> None:
        """Log an info message.

        Args:
            message (str): The message to be logged.
            ranks (List[int]): List of parallel ranks.
        """
        message_prefix = "{}:{} {}".format(*self.__get_call_info())
        self._log("info", message_prefix, ranks)
        self._log("info", message, ranks)

    def warning(self, message: str, ranks: List[int] = None) -> None:
        """Log a warning message.

        Args:
            message (str): The message to be logged.
            ranks (List[int]): List of parallel ranks.
        """
        message_prefix = "{}:{} {}".format(*self.__get_call_info())
        self._log("warning", message_prefix, ranks)
        self._log("warning", message, ranks)

    def debug(self, message: str, ranks: List[int] = None) -> None:
        """Log a debug message.

        Args:
            message (str): The message to be logged.
            ranks (List[int]): List of parallel ranks.
        """
        message_prefix = "{}:{} {}".format(*self.__get_call_info())
        self._log("debug", message_prefix, ranks)
        self._log("debug", message, ranks)

    def error(self, message: str, ranks: List[int] = None) -> None:
        """Log an error message.

        Args:
            message (str): The message to be logged.
            ranks (List[int]): List of parallel ranks.
        """
        message_prefix = "{}:{} {}".format(*self.__get_call_info())
        self._log("error", message_prefix, ranks)
        self._log("error", message, ranks)
