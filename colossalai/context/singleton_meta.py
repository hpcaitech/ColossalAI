import threading


class SingletonMeta(type):
    """
    Thread-safe Singleton Meta with double-checked locking.
    Reference: https://en.wikipedia.org/wiki/Double-checked_locking
    """

    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # First check (without locking) for performance reasons
        if cls not in cls._instances:
            # Acquire a lock before proceeding to the second check
            with cls._lock:
                # Second check with lock held to ensure thread safety
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        else:
            assert (
                len(args) == 0 and len(kwargs) == 0
            ), f"{cls.__name__} is a singleton class and an instance has been created."

        return cls._instances[cls]
