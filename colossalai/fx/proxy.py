import operator
import torch
from torch.fx.proxy import Proxy, Attribute
from typing import List, Union, Any

__all__ = ['ColoProxy']


class ColoProxy(Proxy):
    """
    ColoProxy is a proxy class which uses meta tensor to handle data-dependent control flow. The original torch.fx proxy
    cannot be used to infer the condition statement, with this proxy, torch.fx can still run even with if statements.

    Usage:
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
        assert torch.is_tensor(
            self._meta_data) and self._meta_data.is_meta, f'Meta data is not a meta tensor for {self.node.name}'

    def _assert_has_meta_data(self):
        assert self._meta_data is not None, f'Meta data is not set for {self.node.name}'

    def __len__(self):
        self._assert_has_meta_data()
        return len(self.meta_data)

    def __bool__(self):
        self._assert_has_meta_data()
        return self.meta_data

    def __getattr__(self, k):

        return ColoAttribute(self, k)

    def __setitem__(self, indices, values):
        return self.tracer.create_proxy("call_function", operator.setitem, (self, indices, values), {})

    def __contains__(self, key):
        if self.node.op == "placeholder":
            # this is used to handle like
            # if x in kwargs
            # we don't handle this case for now
            return False
        return super().__contains__(key)


class ColoAttribute(ColoProxy):

    def __init__(self, root, attr: str):
        # this class is copied from torch.fx.Attribute
        # but inherits ColoProxy
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node = None

    @property
    def node(self):
        if self._node is None:
            self._node = self.tracer.create_proxy("call_function", getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy("call_method", self.attr, (self.root,) + args, kwargs)
