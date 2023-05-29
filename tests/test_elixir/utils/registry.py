from collections import OrderedDict
from typing import Callable


class Registry(object):

    def __init__(self) -> None:
        super().__init__()
        self._registry_dict = OrderedDict()

    def register(self, name: str, model_fn: Callable, data_fn: Callable):
        assert name not in self._registry_dict

        model_tuple = (model_fn, data_fn)
        self._registry_dict[name] = model_tuple

    def get(self, name: str):
        return self._registry_dict[name]

    def __iter__(self):
        return iter(self._registry_dict.items())


TEST_MODELS = Registry()

__all__ = [TEST_MODELS]
