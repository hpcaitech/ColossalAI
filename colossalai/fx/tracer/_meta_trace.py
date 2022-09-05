import torch
from torch.fx import Node, Graph
from torch.fx.graph import _Namespace
from torch.utils._pytree import tree_map


def meta_trace(module: torch.nn.Module, *args, **kwargs) -> Graph:
    """Trace forward and backward graph with MetaTensor

    Args:
        module (torch.nn.Module): The target module for tracing.

    Returns:
        graph (torch.fx.Graph): The computation graph.

    Usage:
        >>> import torchvision.models as tm
        >>> model = tm.alexnet()
        >>> graph = meta_trace(model, torch.rand(1000, 3, 224, 224))
        >>> graph.print_tabular()
    """
    graph = Graph()
    namespace = _Namespace()

    class MetaProxy(torch.Tensor):
        """
        A wrapping tensor that hacks `torch.autograd` without patching more `torch.ops.aten` ops.
        """

        _tensor: torch.Tensor
        _node: Node

        __slots__ = ['_tensor', '_node']

        @staticmethod
        def __new__(cls, tensor, placeholder=False, name=None):
            r = torch.Tensor._make_wrapper_subclass(
                cls,
                tensor.size(),
                strides=tensor.stride(),
                storage_offset=tensor.storage_offset(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                device='cpu',
                requires_grad=tensor.requires_grad)    # deceive the frontend for aten selections
            r._tensor = tensor
            if placeholder:
                if name is None:
                    name = 'input'
                r._node = graph.create_node('placeholder',
                                            'placeholder', (graph._root,),
                                            name=namespace.create_name(name, tensor))
            # ...the real tensor is held as an element on the tensor.
            return r

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

            def unwrap(x):
                if isinstance(x, torch.Tensor) and not hasattr(x, '_tensor'):
                    x = MetaProxy(x)
                return x._tensor.to('meta') if isinstance(x, MetaProxy) else x

            def get_node(x):
                if isinstance(x, torch.Tensor) and not hasattr(x, '_node'):
                    x = MetaProxy(x, placeholder=True, name='weight')
                return x if not hasattr(x, '_node') else x._node

            args_node = tree_map(get_node, args)
            kwargs_node = tree_map(get_node, kwargs)
            node = graph.create_node('call_function', func, args_node, kwargs_node)

            args = tree_map(unwrap, args)
            kwargs = tree_map(unwrap, kwargs)

            # run aten for backend=CPU but actually on backend=Meta
            out = func(*args, **kwargs)

            # Now, we want to continue propagating this tensor, so we rewrap Tensors in
            # our custom tensor subclass
            def wrap(x):
                return MetaProxy(x) if isinstance(x, torch.Tensor) and not hasattr(x, '_tensor') else x

            def set_node(x):
                x._node = node

            out = tree_map(wrap, out)
            tree_map(set_node, out)

            return out

    def wrap(x):
        return MetaProxy(x, True) if isinstance(x, torch.Tensor) else x

    args = tree_map(wrap, args)
    kwargs = tree_map(wrap, kwargs)

    module(*args, **kwargs).sum().backward()
    return graph
