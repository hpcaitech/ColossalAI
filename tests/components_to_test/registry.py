#!/usr/bin/env python


class Registry:

    def __init__(self):
        self._registry = dict()

    def register(self, name):
        assert name not in self._registry

        def _regsiter(callable_):
            self._registry[name] = callable_

        return _regsiter

    def get_callable(self, name: str):
        return self._registry[name]

    def __iter__(self):
        self._idx = 0
        self._len = len(self._registry)
        self._names = list(self._registry.keys())
        return self

    def __next__(self):
        if self._idx < self._len:
            key = self._names[self._idx]
            callable_ = self._registry[key]
            self._idx += 1
            return callable_
        else:
            raise StopIteration


non_distributed_component_funcs = Registry()
model_paralle_component_funcs = Registry()

__all__ = ['non_distributed_component_funcs', 'model_paralle_component_funcs']
