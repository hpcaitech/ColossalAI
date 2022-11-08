import torch
from torch.fx import Graph, Node
from torch.utils._pytree import tree_map


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def is_autogradable(x):
    return isinstance(x, torch.Tensor) and x.is_floating_point()


def meta_trace(module: torch.nn.Module, fake_device=None, *args, **kwargs) -> Graph:
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
    namespace = graph._graph_namespace

    class MetaProxy(torch.Tensor):
        """
        A wrapping tensor that hacks `torch.autograd` without patching more `torch.ops.aten` ops.
        """

        _tensor: torch.Tensor
        _node: Node

        __slots__ = ['_tensor', '_node']

        @staticmethod
        def __new__(cls, tensor, fake_device=None, placeholder=False, name=None):
            r = torch.Tensor._make_wrapper_subclass(
                cls,
                tensor.size(),
                strides=tensor.stride(),
                storage_offset=tensor.storage_offset(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                device=fake_device if fake_device is not None else tensor.device,
                requires_grad=tensor.requires_grad)    # deceive the frontend for aten selections
            r._tensor = tensor
            if placeholder:
                if name is None:
                    name = 'input'
                r._node = graph.create_node('placeholder',
                                            'placeholder', (graph._root,),
                                            name=namespace.create_name(name, tensor))
            # ...the real tensor is held as an element on the tensor.
            if not r._tensor.is_meta:
                r._tensor = r._tensor.to(torch.device('meta'))
            return r

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

            def unwrap(x):
                nonlocal fake_device
                if isinstance(x, MetaProxy):
                    fake_device = x.device
                    x = x._tensor
                    # assert not isinstance(x, MetaProxy)
                elif isinstance(x, torch.Tensor):
                    fake_device = x.device
                    x = x.to(torch.device('meta'))
                return x

            def get_node(x):
                if isinstance(x, torch.Tensor) and not hasattr(x, '_node'):
                    x = MetaProxy(x, placeholder=True, name='weight')
                return x if not hasattr(x, '_node') else x._node

            args_node = tree_map(get_node, args)
            kwargs_node = tree_map(get_node, kwargs)
            node = graph.create_node('call_function', func, args_node, kwargs_node)

            if 'device' in kwargs:
                fake_device = kwargs['device']
                kwargs['device'] = torch.device('meta')

            args = tree_map(unwrap, args)
            kwargs = tree_map(unwrap, kwargs)

            # run aten for backend=CPU but actually on backend=Meta
            out = func(*args, **kwargs)

            # Now, we want to continue propagating this tensor, so we rewrap Tensors in
            # our custom tensor subclass
            def wrap(x):
                if isinstance(x, torch.Tensor):
                    nonlocal fake_device
                    if not x.is_meta:
                        x = x.to(torch.device('meta'))
                return MetaProxy(
                    x, fake_device=fake_device) if isinstance(x, torch.Tensor) and not hasattr(x, '_tensor') else x

            def set_node(x):
                x._node = node

            out = tree_map(wrap, out)
            tree_map(set_node, out)

            return out

    def wrap(x):
        return MetaProxy(x, fake_device=fake_device, placeholder=True) if isinstance(x, torch.Tensor) else x

    args = tree_map(wrap, args)
    kwargs = tree_map(wrap, kwargs)

    out = module(*args, **kwargs)

    for tensor in normalize_tuple(out):
        if is_autogradable(tensor) and tensor.requires_grad:
            grad = torch.empty_like(tensor._tensor, device=torch.device('meta')) if isinstance(
                tensor, MetaProxy) else torch.empty_like(tensor, device=torch.device('meta'))
            torch.autograd.backward(tensor,
                                    MetaProxy(grad, fake_device=tensor.device, placeholder=True),
                                    retain_graph=True)
    return graph
