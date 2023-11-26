import operator
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.fx import Node, Proxy
from torch.utils._pytree import tree_map

from colossalai._analyzer._subclasses import MetaTensor

Target = Union[Callable[..., Any], str]


class ColoProxy(Proxy):
    _func_dispatch: Dict[Target, Callable[..., Any]] = {}

    def __init__(self, *args, data=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._meta_data = data

    @property
    def meta_data(self):
        return self._meta_data

    @meta_data.setter
    def meta_data(self, args):
        wrap_fn = lambda x: MetaTensor(x) if isinstance(x, torch.Tensor) else x
        self._meta_data = tree_map(wrap_fn, args)

    @classmethod
    def __torch_function__(cls, orig_method, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if orig_method in cls._func_dispatch:
            impl = cls._func_dispatch.pop(orig_method)  # avoid recursion
            proxy = impl(*args, **kwargs)
            cls._func_dispatch[orig_method] = impl
            return proxy
        else:
            proxy = cls.from_torch_proxy(super().__torch_function__(orig_method, types, args, kwargs))
            unwrap_fn = lambda p: p.meta_data if isinstance(p, ColoProxy) else p
            if proxy.meta_data is None:
                proxy.meta_data = orig_method(*tree_map(unwrap_fn, args), **tree_map(unwrap_fn, kwargs))
            return proxy

    @classmethod
    def from_torch_proxy(cls, proxy: Proxy):
        return cls(proxy.node, proxy.tracer)

    def __repr__(self):
        return f"ColoProxy({self.node.name}, meta_data={self.meta_data})"

    def __len__(self):
        return len(self.meta_data)

    def __int__(self):
        return int(self.meta_data)

    def __index__(self):
        try:
            return int(self.meta_data)
        except:
            return torch.zeros(self.meta_data.shape, dtype=torch.bool).numpy().__index__()

    def __float__(self):
        return float(self.meta_data)

    def __bool__(self):
        return self.meta_data

    def __getattr__(self, k):
        return ColoAttribute(self, k, getattr(self._meta_data, k, None))

    def __setitem__(self, key, value):
        proxy = self.tracer.create_proxy("call_function", operator.setitem, (self, key, value), {})
        proxy.meta_data = self._meta_data
        return proxy

    def __contains__(self, key):
        if self.node.op == "placeholder":
            # this is used to handle like
            # if x in kwargs
            # we don't handle this case for now
            return False
        return super().__contains__(key)

    def __isinstancecheck__(self, type):
        return isinstance(self.meta_data, type)


class ColoAttribute(ColoProxy):
    def __init__(self, root, attr: str, data=None):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._meta_data = data
        self._node: Optional[Node] = None

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy("call_function", getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy("call_method", self.attr, (self.root,) + args, kwargs)

    def __repr__(self):
        return f"ColoAttribute({self.node.name}, attr={self.attr})"
