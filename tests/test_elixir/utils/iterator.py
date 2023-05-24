from abc import ABC, abstractmethod


class TestIterator(ABC):

    def __init__(self, length=10) -> None:
        self.length = length
        self.step = 0

    @abstractmethod
    def generate(self):
        pass

    def __next__(self):
        if self.step < self.length:
            self.step += 1
            return self.generate()
        else:
            raise StopIteration

    def __len__(self):
        return self.length
