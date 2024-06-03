from abc import ABC, abstractmethod
from typing import Any

from coati.experience_maker.base import Experience


class ExperienceBuffer(ABC):
    """Experience buffer base class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
    """

    def __init__(self, sample_batch_size: int, limit: int = 0) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit

    @abstractmethod
    def append(self, experience: Experience) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def sample(self) -> Experience:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        pass

    @abstractmethod
    def collate_fn(self, batch: Any) -> Experience:
        pass
