from abc import ABC, abstractmethod
from typing import Callable


class BaseExtension(ABC):
    @abstractmethod
    def requires_build(self) -> bool:
        pass

    @abstractmethod
    def build(self) -> None:
        pass

    @abstractmethod
    def load(self) -> Callable:
        pass

    def fetch(self) -> Callable:
        if self.requires_build:
            self.build()
        return self.load()
