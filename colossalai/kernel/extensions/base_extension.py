from abc import ABC, abstractmethod
from typing import Callable


class BaseExtension(ABC):
    @abstractmethod
    @property
    def build_completed(self) -> bool:
        pass

    @abstractmethod
    def build(self) -> None:
        pass

    @abstractmethod
    def load(self) -> Callable:
        pass

    def fetch(self) -> Callable:
        if not self.build_completed:
            self.build()
        return self.load()


class CUDAExtension(BaseExtension):
    pass


class TritonExtension(BaseExtension):
    pass


class NPUExtension(BaseExtension):
    pass
