from abc import ABC, abstractmethod
from typing import Dict, Type
from .writer import CheckpointWriter, DiskCheckpointWriter
from .reader import CheckpointReader, DiskCheckpointReader

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
