"""
A class that can be used to calculate the mean of a variable
"""


class AccumulativeMeanVariable:
    def __init__(self):
        self._sum = 0
        self._count = 0

    def add(self, value, count_update=1):
        self._sum += value
        self._count += count_update

    def get(self):
        return self._sum / self._count if self._count > 0 else 0

    def reset(self):
        self._sum = 0
        self._count = 0


class AccumulativeMeanMeter:
    def __init__(self):
        self.variable_dict = {}

    def add(self, name, value, count_update=1):
        if name not in self.variable_dict:
            self.variable_dict[name] = AccumulativeMeanVariable()
        self.variable_dict[name].add(value, count_update=count_update)

    def get(self, name):
        return self.variable_dict[name].get()

    def reset(self):
        for name in self.variable_dict:
            self.variable_dict[name].reset()
