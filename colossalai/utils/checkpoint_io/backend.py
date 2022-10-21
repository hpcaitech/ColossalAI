import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, List, Type

from .reader import CheckpointReader, DiskCheckpointReader
from .writer import CheckpointWriter, DiskCheckpointWriter

_backends: Dict[str, Type['CheckpointIOBackend']] = {}


def register(name: str):
    assert name not in _backends, f'"{name}" is registered'

    def wrapper(cls):
        _backends[name] = cls
        return cls

    return wrapper


def get_backend(name: str) -> 'CheckpointIOBackend':
    assert name in _backends, f'Unsupported backend "{name}"'
    return _backends[name]()


class CheckpointIOBackend(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.temps: List[str] = []

    @abstractmethod
    def get_writer(self,
                   base_name: str,
                   overwrite: bool = False,
                   rank: int = 0,
                   world_size: int = 1) -> CheckpointWriter:
        pass

    @abstractmethod
    def get_reader(self, base_name: str) -> CheckpointReader:
        pass

    @abstractmethod
    def get_temp(self, base_name: str) -> str:
        pass

    @abstractmethod
    def clean_temp(self) -> None:
        pass


@register('disk')
class CheckpointDiskIO(CheckpointIOBackend):

    def get_writer(self,
                   base_name: str,
                   overwrite: bool = False,
                   rank: int = 0,
                   world_size: int = 1) -> CheckpointWriter:
        return DiskCheckpointWriter(base_name, overwrite, rank=rank, world_size=world_size)

    def get_reader(self, base_name: str) -> CheckpointReader:
        return DiskCheckpointReader(base_name)

    def get_temp(self, base_name: str) -> str:
        temp_dir_name = tempfile.mkdtemp(dir=base_name)
        self.temps.append(temp_dir_name)
        return temp_dir_name

    def clean_temp(self) -> None:
        for temp_dir_name in self.temps:
            shutil.rmtree(temp_dir_name)
