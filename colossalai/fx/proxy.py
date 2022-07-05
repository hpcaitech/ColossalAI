import operator
import torch
from torch.fx.proxy import Proxy, Attribute

__all__ = ['ColoProxy']


class ColoProxy(Proxy):
    """
    ColoProxy is a proxy class which uses meta tensor to handle data-dependent control flow. The original torch.fx proxy
    cannot be used to infer the condition statement, with this proxy, torch.fx can still run even with if statements.

    Usage:
        proxy = tracer.create_proxy(...)
        proxy.meta_tensor = torch.empty(4, 2, device='meta')
        print(len(proxy)) # expect output 4

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._meta_tensor = None

    @property
    def meta_tensor(self):
        return self._meta_tensor

    @meta_tensor.setter
    def meta_tensor(self, tensor: torch.Tensor):
        assert tensor is None or tensor.is_meta, 'Expected to receive a meta tensor, but got a non-meta tensor'
        self._meta_tensor = tensor

    @property
    def has_meta_tensor(self):
        return self.meta_tensor is not None

    def _assert_has_meta(self):
        assert self.has_meta_tensor, f'Meta tensor is not set for {self.node.name}'

    @property
    def device(self):
        # Hack so we can track when devices are used. During meta-tensor propagation,
        # replace these values with a constant 'meta'
        return MetaDeviceAttribute(self, "device")

    @property
    def dtype(self):
        self._assert_has_meta()
        return self.meta_tensor.dtype

    @property
    def shape(self):
        self._assert_has_meta()
        return self.meta_tensor.shape

    def dim(self):
        self._assert_has_meta()
        return self.meta_tensor.dim()

    def size(self, dim: int = None):
        self._assert_has_meta()
        if dim:
            return self.meta_tensor.size(dim=dim)
        else:
            # size(dim=None) will trigger runtime error for meta tensor
            return self.meta_tensor.size()

    def __len__(self):
        self._assert_has_meta()
        return len(self.meta_tensor)

    def __bool__(self):
        self._assert_has_meta()
        return self.meta_tensor

    def __getattr__(self, k):
        if k == "metadata":
            return self.meta_tensor
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return Attribute(self, k)

    def __setitem__(self, indices, values):
        return self.tracer.create_proxy("call_function", operator.setitem, (self, indices, values), {})


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
