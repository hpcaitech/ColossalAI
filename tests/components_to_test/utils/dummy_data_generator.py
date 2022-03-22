from abc import ABC, abstractmethod


class DummyDataGenerator(ABC):

    def __init__(self, length=10):
        self.length = length

    @abstractmethod
    def generate(self):
        pass

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        if self.step < self.length:
            self.step += 1
            return self.generate()
        else:
            raise StopIteration

    def __len__(self):
        return self.length
