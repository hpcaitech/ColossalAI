from typing import Any

import torch
from torch.fx.proxy import Proxy

from colossalai.fx.tracer.meta_patch import meta_patched_function

__all__ = ["ColoProxy"]


class ColoProxy(Proxy):
    """
    ColoProxy is a proxy class which uses meta tensor to handle data-dependent control flow. The original torch.fx proxy
    cannot be used to infer the condition statement, with this proxy, torch.fx can still run even with if statements.

    Example::

        proxy = tracer.create_proxy(...)
        proxy.meta_data = torch.empty(4, 2, device='meta')
        print(len(proxy)) # expect output 4

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node._meta_data = None

    @property
    def meta_data(self):
        return self.node._meta_data

    @meta_data.setter
    def meta_data(self, data: Any):
        self.node._meta_data = data

    @property
    def has_meta_data(self):
        return self._meta_data is not None

    def _assert_meta_data_is_tensor(self):
        assert (
            torch.is_tensor(self._meta_data) and self._meta_data.is_meta
        ), f"Meta data is not a meta tensor for {self.node.name}"

    def _assert_has_meta_data(self):
        assert self._meta_data is not None, f"Meta data is not set for {self.node.name}"

    def __len__(self):
        self._assert_has_meta_data()
        return len(self.meta_data)

    def __int__(self):
        self._assert_has_meta_data()
        return int(self.meta_data)

    def __float__(self):
        self._assert_has_meta_data()
        return float(self.meta_data)

    def __bool__(self):
        self._assert_has_meta_data()
        return self.meta_data

    def __getattr__(self, k):
        return ColoAttribute(self, k)

    def __contains__(self, key):
        if self.node.op == "placeholder":
            # this is used to handle like
            # if x in kwargs
            # we don't handle this case for now
            return False
        return super().__contains__(key)


def extract_meta(*args, **kwargs):
    """
    This function is copied from _tracer_utils.py to avoid circular import issue.
    """

    def _convert(val):
        if isinstance(val, ColoProxy):
            return val.meta_data
        elif isinstance(val, (list, tuple)):
            return type(val)([_convert(ele) for ele in val])
        return val

    new_args = [_convert(val) for val in args]
    new_kwargs = {k: _convert(v) for k, v in kwargs.items()}
    return new_args, new_kwargs


class ColoAttribute(ColoProxy):
    def __init__(self, root, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node = None

    @property
    def node(self):
        if self._node is None:
            proxy = self.tracer.create_proxy("call_function", getattr, (self.root, self.attr), {})
            if not isinstance(proxy, ColoProxy):
                meta_args, meta_kwargs = extract_meta(*(self.root, self.attr))
                meta_out = getattr(*meta_args, **meta_kwargs)
                proxy = ColoProxy(proxy.node)
                proxy.meta_data = meta_out
            self._node = proxy.node

        return self._node

    def __call__(self, *args, **kwargs):
        proxy = self.tracer.create_proxy("call_method", self.attr, (self.root,) + args, kwargs)
        if not isinstance(proxy, ColoProxy):
            meta_args, meta_kwargs = extract_meta(*((self.root,) + args), **kwargs)
            method = getattr(meta_args[0].__class__, self.attr)
            if meta_patched_function.has(method):
                meta_target = meta_patched_function.get(method)
            elif meta_patched_function.has(method.__name__):
                meta_target = meta_patched_function.get(method.__name__)
            else:
                meta_target = method
            meta_out = meta_target(*meta_args, **meta_kwargs)
            proxy = ColoProxy(proxy.node)
            proxy.meta_data = meta_out
        return proxy
