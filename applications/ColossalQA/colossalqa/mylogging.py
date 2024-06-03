"""
Class for logging with extra control for debugging
"""
import logging


class ColossalQALogger:
    """This is a distributed event logger class essentially based on :class:`logging`.

    Args:
        name (str): The name of the logger.

    Note:
        Logging types: ``info``, ``warning``, ``debug`` and ``error``
    """

    __instances = dict()

    def __init__(self, name):
        if name in ColossalQALogger.__instances:
            raise ValueError("Logger with the same name has been created")
        else:
            self._name = name
            self._logger = logging.getLogger(name)

            ColossalQALogger.__instances[name] = self

    @staticmethod
    def get_instance(name: str):
        """Get the unique single logger instance based on name.

        Args:
            name (str): The name of the logger.

        Returns:
            DistributedLogger: A DistributedLogger object
        """
        if name in ColossalQALogger.__instances:
            return ColossalQALogger.__instances[name]
        else:
            logger = ColossalQALogger(name=name)
            return logger

    def info(self, message: str, verbose: bool = False) -> None:
        """Log an info message.

        Args:
            message (str): The message to be logged.
            verbose (bool): Whether to print the message to stdout.
        """
        if verbose:
            logging.basicConfig(level=logging.INFO)
            self._logger.info(message)

    def warning(self, message: str, verbose: bool = False) -> None:
        """Log a warning message.

        Args:
            message (str): The message to be logged.
            verbose (bool): Whether to print the message to stdout.
        """
        if verbose:
            self._logger.warning(message)

    def debug(self, message: str, verbose: bool = False) -> None:
        """Log a debug message.

        Args:
            message (str): The message to be logged.
            verbose (bool): Whether to print the message to stdout.
        """
        if verbose:
            self._logger.debug(message)

    def error(self, message: str) -> None:
        """Log an error message.

        Args:
            message (str): The message to be logged.
        """
        self._logger.error(message)


def get_logger(name: str = None, level=logging.INFO) -> ColossalQALogger:
    """
    Get the logger by name, if name is None, return the default logger
    """
    if name:
        logger = ColossalQALogger.get_instance(name=name)
    else:
        logger = ColossalQALogger.get_instance(name="colossalqa")
    return logger
