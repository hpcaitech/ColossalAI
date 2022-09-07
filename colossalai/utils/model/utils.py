import torch
import functools
from typing import Optional


def substitute_init_recursively(cls, func, visited: set):
    for subcls in cls.__subclasses__():
        substitute_init_recursively(subcls, func, visited)
        if subcls not in visited:
            func(subcls)
            visited.add(subcls)


def call_to_str(base, *args, **kwargs):
    """Construct a string representation of a call.

    Args:
        base (str): name of the call
        args (tuple, optional): args to ``base``
        kwargs (dict, optional): kwargs supplied to ``base``

    Returns:
        str: A string representation of base(*args, **kwargs)
    """
    name = f'{base}('
    if args:
        name += ', '.join(repr(arg) for arg in args)
        if kwargs:
            name += ', '
    if kwargs:
        name += ', '.join(f'{key}={repr(arg)}' for key, arg in kwargs.items())
    name += ')'
    return name


class InsertPostInitMethodToModuleSubClasses(object):

    def __init__(self, default_dtype: Optional[torch.dtype] = None):
        self._old_default_dtype = None
        self._default_dtype = default_dtype

    def __enter__(self):
        r"""
        Enter the context scope.
        """
        if self._default_dtype is not None:
            self._old_default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(self._default_dtype)

        def preprocess_after(f):

            @functools.wraps(f)
            def wrapper(module: torch.nn.Module, *args, **kwargs):
                f(module, *args, **kwargs)
                self._post_init_method(module, *args, **kwargs)

            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = preprocess_after(cls.__init__)

        # The function is called during init subclass.
        def _init_subclass(cls, **kwargs):
            cls.__init__ = preprocess_after(cls.__init__)

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        # Excution self._post_init_method after the default init function.
        substitute_init_recursively(torch.nn.modules.module.Module, _enable_class, set())

        # holding on to the current __init__subclass__ for exit
        torch.nn.modules.module.Module._old_init_subclass = (torch.nn.modules.module.Module.__init_subclass__)
        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)

        self._pre_context_exec()
        return self

    def __exit__(self, exc_type, exc_value, traceback):

        if self._default_dtype is not None:
            torch.set_default_dtype(self._old_default_dtype)

        def _disable_class(cls):
            if not hasattr(cls, '_old_init'):
                raise AttributeError(
                    f"_old_init is not found in the {cls.__name__}, please make sure that you have imported {cls.__name__} before entering the context."
                )
            cls.__init__ = cls._old_init

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        substitute_init_recursively(torch.nn.modules.module.Module, _disable_class, set())

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = (torch.nn.modules.module.Module._old_init_subclass)

        self._post_context_exec()
        # Now that we cleaned up the metaclass injection, raise the exception.
        if exc_type is not None:
            return False

    # To be implemented by inheriting classes
    def _post_init_method(self, module, *args, **kwargs):
        pass

    def _pre_context_exec(self):
        pass

    def _post_context_exec(self):
        pass
