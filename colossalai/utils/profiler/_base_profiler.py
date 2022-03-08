from abc import ABC, abstractmethod
from pathlib import Path


class BaseProfiler(ABC):

    @abstractmethod
    def enable(self):
        pass

    @abstractmethod
    def disable(self):
        pass

    @abstractmethod
    def to_tensorboard(self, writer):
        pass

    @abstractmethod
    def to_file(self, filename: Path):
        pass

    @abstractmethod
    def show(self):
        pass

    def get_lastest(self) -> dict:
        return None

    def get_avg(self) -> dict:
        raise None
