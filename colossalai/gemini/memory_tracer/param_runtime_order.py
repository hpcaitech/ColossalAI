from abc import ABC

import torch


class ParamGenerator(ABC):

    def append(self, param: torch.nn.Parameter):
        pass

    def generate(self):
        pass

    def clear(self):
        pass


class OrderedParamGenerator(ParamGenerator):
    """OrderedParamGenerator

    Contain the order of parameters visited during runtime.
    """

    def __init__(self) -> None:
        self.param_visited_order = []

    def append(self, param: torch.nn.Parameter):
        self.param_visited_order.append(param)

    def generate(self):
        visited_set = set()
        for p in self.param_visited_order:
            if p not in visited_set:
                yield p
            visited_set.add(p)
        del visited_set

    def is_empty(self):
        return len(self.param_visited_order) == 0

    def clear(self):
        self.param_visited_order = []
