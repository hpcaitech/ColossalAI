from abc import ABC, abstractmethod
import torch


class BaseOpHook(ABC):
    """This class allows users to add customized operations
    before and after the execution of a PyTorch submodule"""
    def __init__(self):
        pass

    @abstractmethod
    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        pass

    @abstractmethod
    def post_fwd_exec(self, module: torch.nn.Module, *args):
        pass

    @abstractmethod
    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        pass

    @abstractmethod
    def post_bwd_exec(self, module: torch.nn.Module, input):
        pass

    @abstractmethod
    def post_iter(self):
        pass
