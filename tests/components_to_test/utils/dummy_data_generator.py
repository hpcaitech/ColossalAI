from abc import ABC, abstractmethod


class DummyDataGenerator(ABC):

    @abstractmethod
    def generate(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate()
