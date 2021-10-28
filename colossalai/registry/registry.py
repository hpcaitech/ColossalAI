#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from types import ModuleType
from typing import List


class Registry:
    """This is a registry class used to register classes and modules so that a universal 
    object builder can be enabled.

    :param name: The name of the registry
    :type name: str
    :param third_party_library: List of third party libraries which are used in the 
        initialization of the register module
    :type third_party_library: list, optional
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

        :param module_class: The module to be registered
        :type module_class: class
        :raises AssertionError: Raises an AssertionError if the module has already been 
            registered before
        :return: The module to be registered, so as to use it normally if via importing
        :rtype: class
        """
        module_name = module_class.__name__
        assert module_name not in self._registry
        self._registry[module_name] = module_class

        # return so as to use it normally if via importing
        return module_class

    def get_module(self, module_name: str):
        """Retrieves a module with name `module_name` and returns the module if it has 
        already been registered before.

        :param module_name: The name of the module to be retrieved
        :type module_name: str
        :raises NameError: Raises a NameError if the module to be retrieved has neither been 
            registered directly nor as third party modules before
        :return: The retrieved module or None
        :rtype: :class:`object`
        """
        if module_name in self._registry:
            return self._registry[module_name]
        elif self._third_party_lib is not None:
            for lib in self._third_party_lib:
                if hasattr(lib, module_name):
                    return getattr(lib, module_name)
            raise NameError(f'Module {module_name} not found in the registry {self.name}')

    def has(self, module_name: str):
        """Searches for a module with name `module_name` and returns a boolean value indicating
        whether the module has been registered directly or as third party modules before.

        :param module_name: The name of the module to be searched for
        :type module_name: str
        :return: A boolean value indicating whether the module has been registered directly or
            as third party modules before
        :rtype: bool
        """
        found_flag = module_name in self._registry

        if self._third_party_lib:
            for lib in self._third_party_lib:
                if hasattr(lib, module_name):
                    found_flag = True
                    break

        return found_flag
