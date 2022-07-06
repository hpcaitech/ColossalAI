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
        self._meta_data = None

    @property
    def meta_data(self):
        return self._meta_data

    @meta_data.setter
    def meta_data(self, data: Any):
        self._meta_data = data

    @property
    def has_meta_data(self):
        return self._meta_data is not None

    def _assert_meta_data_is_tensor(self):
        assert torch.is_tensor(
            self._meta_data) and self._meta_data.is_meta, f'Meta data is not a meta tensor for {self.node.name}'

    def _assert_has_meta_data(self):
        assert self._meta_data is not None, f'Meta data is not set for {self.node.name}'

    @property
    def device(self):
        # Hack so we can track when devices are used. During meta-tensor propagation,
        # replace these values with a constant 'meta'
        return MetaDeviceAttribute(self, "device")

    @property
    def dtype(self):
        self._assert_meta_data_is_tensor()
        return self.meta_data.dtype

    @property
    def shape(self):
        self._assert_meta_data_is_tensor()
        return self.meta_data.shape

    def dim(self):
        self._assert_meta_data_is_tensor()
        return self.meta_data.dim()

    def size(self, dim: int = None):
        self._assert_meta_data_is_tensor()
        if dim is not None:
            return self.meta_data.size(dim=dim)
        else:
            # size(dim=None) will trigger runtime error for meta tensor
            return self.meta_data.size()

    def __len__(self):
        self._assert_has_meta_data()
        return len(self.meta_data)

    def __bool__(self):
        self._assert_has_meta_data()
        return self.meta_data

    def __getattr__(self, k):
        if k == "meta_data":
            return self.__getattribute__(k)
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return Attribute(self, k)

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


class MetaDeviceAttribute(ColoAttribute):
    pass
