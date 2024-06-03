from collections import OrderedDict
from contextlib import contextmanager
from threading import Lock
from typing import Any, Dict, Hashable, List


class MissCacheError(Exception):
    pass


class ListCache:
    def __init__(self, cache_size: int, list_size: int, fixed_keys: List[Hashable] = []) -> None:
        """Cache a list of values. The fixed keys won't be removed. For other keys, LRU is applied.
        When the value list is not full, a cache miss occurs. Otherwise, a cache hit occurs. Redundant values will be removed.

        Args:
            cache_size (int): Max size for LRU cache.
            list_size (int): Value list size.
            fixed_keys (List[Hashable], optional): The keys which won't be removed. Defaults to [].
        """
        self.cache_size = cache_size
        self.list_size = list_size
        self.cache: OrderedDict[Hashable, List[Any]] = OrderedDict()
        self.fixed_cache: Dict[Hashable, List[Any]] = {}
        for key in fixed_keys:
            self.fixed_cache[key] = []
        self._lock = Lock()

    def get(self, key: Hashable) -> List[Any]:
        with self.lock():
            if key in self.fixed_cache:
                l = self.fixed_cache[key]
                if len(l) >= self.list_size:
                    return l
            elif key in self.cache:
                self.cache.move_to_end(key)
                l = self.cache[key]
                if len(l) >= self.list_size:
                    return l
        raise MissCacheError()

    def add(self, key: Hashable, value: Any) -> None:
        with self.lock():
            if key in self.fixed_cache:
                l = self.fixed_cache[key]
                if len(l) < self.list_size and value not in l:
                    l.append(value)
            elif key in self.cache:
                self.cache.move_to_end(key)
                l = self.cache[key]
                if len(l) < self.list_size and value not in l:
                    l.append(value)
            else:
                if len(self.cache) >= self.cache_size:
                    self.cache.popitem(last=False)
                self.cache[key] = [value]

    @contextmanager
    def lock(self):
        try:
            self._lock.acquire()
            yield
        finally:
            self._lock.release()
