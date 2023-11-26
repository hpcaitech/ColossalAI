#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from types import ModuleType
from typing import List


class Registry:
    """This is a registry class used to register classes and modules so that a universal
    object builder can be enabled.

    Args:
        name (str): The name of the registry .
        third_party_library (list, optional):
            List of third party libraries which are used in the initialization of the register module.
    """

    def __init__(self, name: str, third_party_library: List[ModuleType] = None):
        self._name = name
        self._registry = dict()
        self._third_party_lib = third_party_library

    @property
    def name(self):
        return self._name

    def register_module(self, module_class):
        """Registers a module represented in `module_class`.

        Args:
            module_class (class): The module to be registered.
        Returns:
            class: The module to be registered, so as to use it normally if via importing.
        Raises:
            AssertionError: Raises an AssertionError if the module has already been registered before.
        """
        module_name = module_class.__name__
        assert module_name not in self._registry, f"{module_name} not found in {self.name}"
        self._registry[module_name] = module_class

        # return so as to use it normally if via importing
        return module_class

    def get_module(self, module_name: str):
        """Retrieves a module with name `module_name` and returns the module if it has
        already been registered before.

        Args:
            module_name (str): The name of the module to be retrieved.
        Returns:
            :class:`object`: The retrieved module or None.
        Raises:
            NameError: Raises a NameError if the module to be retrieved has neither been
            registered directly nor as third party modules before.
        """
        if module_name in self._registry:
            return self._registry[module_name]
        elif self._third_party_lib is not None:
            for lib in self._third_party_lib:
                if hasattr(lib, module_name):
                    return getattr(lib, module_name)
            raise NameError(f"Module {module_name} not found in the registry {self.name}")

    def has(self, module_name: str):
        """Searches for a module with name `module_name` and returns a boolean value indicating
        whether the module has been registered directly or as third party modules before.

        Args:
            module_name (str): The name of the module to be searched for.
        Returns:
            bool: A boolean value indicating whether the module has been registered directly or
            as third party modules before.
        """
        found_flag = module_name in self._registry

        if self._third_party_lib:
            for lib in self._third_party_lib:
                if hasattr(lib, module_name):
                    found_flag = True
                    break

        return found_flag
