from abc import ABC, abstractmethod


class ProfilerExtension(ABC):
    @abstractmethod
    def prepare_trace(self):
        pass

    @abstractmethod
    def start_trace(self):
        pass

    @abstractmethod
    def stop_trace(self):
        pass

    @abstractmethod
    def extend_chrome_trace(self, trace: dict) -> dict:
        pass
