import threading

"""
Synchronization decorator
"""


def synchronized(lock):
    def wrap(func):
        def synchronized_func(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)

        return synchronized_func

    return wrap


class SingletonMeta(type):
    """
    Thread-safe Singleton Meta, and we use double-checked locking to ensure no degradation on performance
    Reference: https://en.wikipedia.org/wiki/Double-checked_locking
    """

    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = cls._locked_call(*args, **kwargs)
            cls._instances[cls] = instance
        else:
            assert (
                len(args) == 0 and len(kwargs) == 0
            ), f"{cls.__name__} is a singleton class and a instance has been created."
        return cls._instances[cls]

    @synchronized(_lock)
    def _locked_call(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
