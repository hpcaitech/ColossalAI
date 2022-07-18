#!/usr/bin/env python
"""
tracer.py: 
    Implemented a tracer which supports control flow and user-defined meta arguments.
    The implementation is partly inspired HuggingFace's fx tracer
"""
import enum
import inspect
import functools
from colossalai.fx.tracer.meta_patch import meta_patched_module
import torch
import torch.nn as nn
from torch import Tensor
from torch.fx import Tracer
from torch.fx.graph import Graph
from torch.fx.proxy import Proxy, ParameterProxy
from ..proxy import ColoProxy
from typing import Optional, Dict, Any
from ._tracer_utils import is_element_in_list, extract_meta
from .meta_patch import meta_patched_function, meta_patched_module

__all__ = ['ColoTracer']


class TracerType(enum.Enum):
    DEFAULT = 1
    META = 2


class ColoTracer(Tracer):
    """
    ColoTracer is a symbolic tracer designed to support dynamic control flow by using meta tensors for the `colossalai.fx` module.
    This tracer is initialized in the same way as the original torch.fx.Tracer.

    Usage:
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 10)
                self.linear2 = nn.Linear(10, 10)

            def forward(self, x, y):
                x1 = self.linear1(x)
                y1 = self.linear2(y)

                if x1.dim() == 2:
                    return x1 + y1
                else:
                    return x1 - y1

        model = Model()
        tracer = ColoTracer()
        graph = tracer.trace(model, concrete_args={'y': torch.rand(4, 10)}, meta_args={'x': torch.rand(4, 10, device='meta')})
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracer_type = TracerType.META
        self.proxy_cls = ColoProxy

    # Feature flag for proxying accesses to buffer values
    proxy_buffer_attributes: bool = True

    _TORCH_METHODS_TO_PATCH = ["arange", "zeros", "ones", "full", "full_like", "eye", "empty", "tensor"]

    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None) -> ColoProxy:
        """
        Create a proxy for different kinds of operations.
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)

        if self.tracer_type == TracerType.DEFAULT:
            # since meta_args is not given
            # we just fall back to the original torch.fx.Tracer
            return proxy

        proxy: ColoProxy

        if kind == "placeholder" and target in self.meta_args and self.meta_args[target].is_meta:
            proxy.meta_data = self.meta_args[target]
            return proxy

        if target in self.orig_torch_tensor_methods:
            # NOTE: tensor constructors in PyTorch define the `device` argument as
            # *kwargs-only*. That is why this works. If you add methods to
            # _TORCH_METHODS_TO_PATCH that do not define `device` as kwarg-only,
            # this will break and you will likely see issues where we cannot infer
            # the size of the output.
            if "device" in kwargs:
                kwargs["device"] = "meta"

        try:
            args_metas, kwargs_metas = extract_meta(*args, **kwargs)

            if kind == "call_function":
                # fetch patched function
                if meta_patched_function.has(target):
                    meta_target = meta_patched_function.get(target)
                elif meta_patched_function.has(target.__name__):
                    # use name for some builtin op like @ (matmul)
                    meta_target = meta_patched_function.get(target.__name__)
                else:
                    meta_target = target

                meta_out = meta_target(*args_metas, **kwargs_metas)
                if isinstance(meta_out, torch.Tensor):
                    meta_out = meta_out.to(device="meta")
            elif kind == "call_method":
                method = getattr(args_metas[0].__class__, target)

                # fetch patched method
                if meta_patched_function.has(method):
                    meta_target = meta_patched_function.get(method)
                else:
                    meta_target = method

                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == "call_module":
                if not hasattr(self, "orig_forward"):
                    raise AttributeError(f"{self} does not have an attribute called orig_forward")
                self._disable_module_getattr = True
                try:
                    mod = self.root.get_submodule(target)
                    mod_type = type(mod)
                    if meta_patched_module.has(mod_type):
                        meta_out = meta_patched_module.get(mod_type)(mod, *args_metas, **kwargs_metas)
                    else:
                        meta_out = self.orig_forward(*args_metas, **kwargs_metas)
                finally:
                    self._disable_module_getattr = False
            elif kind == "get_attr":
                self._disable_module_getattr = True
                try:
                    attr_itr = self.root
                    atoms = target.split(".")
                    for atom in atoms:
                        attr_itr = getattr(attr_itr, atom)
                    if isinstance(attr_itr, torch.Tensor):
                        meta_out = attr_itr.to(device="meta")
                    else:
                        meta_out = attr_itr
                finally:
                    self._disable_module_getattr = False
            else:
                return proxy

            if not isinstance(proxy, Proxy):
                raise ValueError("Don't support composite output yet")
            proxy.meta_data = meta_out
        except Exception as e:
            raise RuntimeError(f"Could not compute metadata for {kind} target {target}: {e}")
        return proxy

    def _module_getattr(self, attr, attr_val, parameter_proxy_cache):
        if getattr(self, "_disable_module_getattr", False):
            return attr_val
        else:
            # return super()._module_getattr(attr, attr_val, parameter_proxy_cache)
            def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
                for n, p in collection_to_search:
                    if attr_val is p:
                        if n not in parameter_proxy_cache:
                            kwargs = {}
                            if "proxy_factory_fn" in inspect.signature(self.create_proxy).parameters:
                                kwargs["proxy_factory_fn"] = (None if not self.param_shapes_constant else
                                                              lambda node: ParameterProxy(self, node, n, attr_val))
                            val_proxy = self.create_proxy("get_attr", n, (), {}, **kwargs)    # type: ignore[arg-type]
                            parameter_proxy_cache[n] = val_proxy
                        return parameter_proxy_cache[n]
                return None

            if isinstance(attr_val, torch.nn.Parameter):
                maybe_parameter_proxy = maybe_get_proxy_for_attr(attr_val, self.root.named_parameters(),
                                                                 parameter_proxy_cache)
                if maybe_parameter_proxy is not None:
                    return maybe_parameter_proxy

            if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
                maybe_buffer_proxy = maybe_get_proxy_for_attr(attr_val, self.root.named_buffers(),
                                                              parameter_proxy_cache)
                if maybe_buffer_proxy is not None:
                    return maybe_buffer_proxy

            return attr_val

    def call_module(self, m, forward, args, kwargs):
        self.orig_forward = forward
        module_qualified_name = self.path_of_module(m)

        # a leaf module is the torch.nn.Module subclasses starting with `torch.nn`
        # which means customized modules are not leaf module by default
        # if a customized or third-party module like apex.normalization.FusedRMSNorm is patched,
        # we should treat it as leaf module as well
        if meta_patched_module.has(m.__class__) or self.is_leaf_module(m, module_qualified_name):
            return self.create_proxy('call_module', module_qualified_name, args, kwargs)
        else:
            return forward(*args, **kwargs)

    def proxy(self, node) -> Proxy:
        """
        Returns a ColoProxy object.
        """
        return self.proxy_cls(node, self)

    def _configure_tracer_type(self, tracer_type: TracerType):
        if tracer_type == TracerType.DEFAULT:
            self.proxy_cls = Proxy
            self.tracer_type = TracerType.DEFAULT
        elif tracer_type == TracerType.META:
            self.proxy_cls = ColoProxy
            self.tracer_type = TracerType.META
        else:
            raise ValueError(f"Unrecognised tracer type {tracer_type}")

    def trace(self,
              root: nn.Module,
              concrete_args: Optional[Dict[str, Tensor]] = None,
              meta_args: Optional[Dict[str, Tensor]] = None) -> Graph:
        """
        Trace the forward computation graph using `torch.fx.Tracer`. This tracer enables data-dependent control flow.

        Args:
            root (nn.Module): a `nn.Module` object to trace the computation graph
            meta_args (Optional[Dict[str, Tensor]]): the meta tensor arguments used to trace the computation graph. 
                These arguments are the sample data fed to the model during actual computation, but just converted to meta tensors.
            concrete_args (Optional[Dict[str, Tensor]]): the concrete arguments that should not be treated as Proxies.
        """
        if meta_args is None:
            meta_args = {}

        if concrete_args is None:
            concrete_args = {}

        if len(meta_args) == 0:
            self._configure_tracer_type(TracerType.DEFAULT)
        else:
            self._configure_tracer_type(TracerType.META)

        # check concrete and meta args have valid names
        sig = inspect.signature(root.forward)
        sig_names = set(sig.parameters.keys())
        meta_arg_names = set(meta_args.keys())

        # update concrete args with default values
        non_meta_arg_names = sig_names - meta_arg_names
        for k, v in sig.parameters.items():
            if k in non_meta_arg_names and \
                k not in concrete_args and \
                v.default is not inspect.Parameter.empty:
                concrete_args[k] = v.default

        # get non concrete arg names
        concrete_arg_names = set(concrete_args.keys())
        non_concrete_arg_names = sig_names - concrete_arg_names

        def _check_arg_name_valid(names):
            success, element = is_element_in_list(names, sig_names)
            if not success:
                raise KeyError(
                    f"argument {element} is not found in the signature of {root.__class__.__name__}'s forward function")

        _check_arg_name_valid(meta_arg_names)
        _check_arg_name_valid(concrete_arg_names)

        # assign as attributed for late reference
        def _check_kwargs(kwargs, should_be_meta: bool):
            for k, v in kwargs.items():
                if not should_be_meta:
                    assert not torch.is_tensor(v) or not v.is_meta, \
                        f'Expected the {k} not to be a meta tensor, please check the args passed to the tracer'
                else:
                    assert v.is_meta == should_be_meta, \
                        f'Expected the is_meta attribute of {k} to be {should_be_meta}, but got {v.is_meta}, please check the args passed to the tracer'

        _check_kwargs(concrete_args, should_be_meta=False)
        _check_kwargs(meta_args, should_be_meta=True)

        self.concrete_args = concrete_args
        self.meta_args = meta_args

        self.patched_torch_tensor_methods = {}
        if self.tracer_type == TracerType.META:
            # wrap the torch tensor constructing methods so that they are captured in the graph
            self.patched_torch_tensor_methods = {
                target: wrap_tensor_constructor_method(getattr(torch, target))
                for target in self._TORCH_METHODS_TO_PATCH
            }

            # patch these methods to replace their original use
            for name, (wrapper, orig) in self.patched_torch_tensor_methods.items():
                setattr(torch, name, wrapper)

            # cache these methods so that we can detect whether a method call
            # should be patched during tracing
            self.orig_torch_tensor_methods = [val[1] for val in self.patched_torch_tensor_methods.values()]

        try:
            self.graph = super().trace(root, concrete_args=concrete_args)
        finally:
            # recover the patched methods
            for name, (_, orig) in self.patched_torch_tensor_methods.items():
                setattr(torch, name, orig)

        if self.tracer_type == TracerType.DEFAULT:
            return self.graph

        # This is necessary because concrete args are added as input to the traced module since
        # https://github.com/pytorch/pytorch/pull/55888.
        for node in self.graph.nodes:
            if node.op == "placeholder":
                # Removing default values for inputs as the forward pass will fail with them.
                if node.target in non_concrete_arg_names:
                    node.args = ()
                    # Without this, torch.jit.script fails because the inputs type is Optional[torch.Tensor].
                    # It cannot infer on the attributes and methods the input should have, and fails.
                    node.type = torch.Tensor
                # It is a concrete arg so it is not used and should be removed.
                else:
                    if hasattr(torch.fx._symbolic_trace, "_assert_is_none"):
                        # Newer versions of torch.fx emit an assert statement
                        # for concrete arguments; delete those before we delete
                        # the concrete arg.
                        to_delete = []
                        for user in node.users:
                            if user.target == torch.fx._symbolic_trace._assert_is_none:
                                to_delete.append(user)
                        for user in to_delete:
                            self.graph.erase_node(user)

                    self.graph.erase_node(node)

            # TODO: solves GraphModule creation.
            # Without this, return type annotation "Tuple" is causing code execution failure.
            if node.op == "output":
                node.type = None

        return self.graph


def wrap_tensor_constructor_method(target):

    def look_for_proxy(*args, **kwargs):
        # find in pos vars
        for arg in args:
            if isinstance(arg, Proxy):
                return arg

        # find in keyword vars
        for k, v in kwargs.items():
            if isinstance(v, Proxy):
                return v
        return None

    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        proxy = look_for_proxy(*args, **kwargs)

        if proxy is not None:
            # if the arg is a proxy, then need to record this function called on this proxy
            # e.g. torch.ones(size) where size is an input proxy
            return proxy.tracer.create_proxy("call_function", target, args, kwargs)
        else:
            # this is called directly when the inputs do not contain proxy
            # e.g. torch.ones(4) where the input is static
            return target(*args, **kwargs)

    return wrapper, target
