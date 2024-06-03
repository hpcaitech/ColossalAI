import functools
import inspect
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.fx import Graph, Node, Proxy, Tracer
from torch.utils._pytree import tree_map

from colossalai._analyzer._subclasses import _TensorPropertyMethod, _TorchFactoryMethod

from ..node_util import MetaInfo
from .proxy import ColoProxy

Target = Union[Callable[..., Any], str]


def _truncate_suffix(s: str):
    import re

    # FIXME: don't know why but torch.fx always gets a suffix like '_1' in the name
    return re.sub(r"_\d+$", "", s)


def register_tracer_impl(func: Callable[..., Any], name: Optional[str] = "_custom_impl"):
    def wrapper(impl):
        assert hasattr(ColoTracer, name), f"Cannot register {func.__name__} in ColoTracer.{name}"
        getattr(ColoTracer, name)[func] = impl
        return impl

    return wrapper


def register_leaf_module_impl(module: nn.Module):
    def wrapper(impl):
        ColoTracer._custom_leaf_module_impl[module] = impl
        return impl

    return wrapper


def register_leaf_module(module: nn.Module):
    ColoTracer._custom_leaf_module.add(module)


def register_non_leaf_module(module: nn.Module):
    ColoTracer._custom_non_leaf_module.add(module)


class ColoTracer(Tracer):
    _custom_leaf_module: Set[Type[nn.Module]] = set()
    _custom_leaf_module_impl: Dict[Type[nn.Module], Callable[..., Any]] = {}
    _custom_non_leaf_module: Set[Type[nn.Module]] = set()
    _custom_impl: Dict[Callable[..., Any], Callable[..., Any]] = {}
    _bias_addition_impl: Dict[Callable[..., Any], Callable[..., Any]] = {}
    _bias_addition_module = [
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
    ]

    def __init__(self, trace_act_ckpt: bool = False, bias_addition_split: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_module_getattr = False
        self.proxy_buffer_attributes = True

        # whether the tracer will record the usage of torch.utils.checkpoint
        self.trace_act_ckpt = trace_act_ckpt
        self.ckpt_regions = []
        self.ckpt_idx = 0

        self.mod_dir = ""

        # whether the tracer should split the bias_add ops into two ops
        self.bias_addition_split = bias_addition_split

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # if bias-addiction split is enabled, and module has bias, then it is not a leaf module
        # we will enter the module and split the bias-addition ops
        if self.bias_addition_split and type(m) in self._bias_addition_module and m.bias is not None:
            return False
        # user can specify which modules are leaf modules and which are not
        return type(m) not in self._custom_non_leaf_module and (
            type(m) in self._custom_leaf_module or super().is_leaf_module(m, module_qualified_name)
        )

    def call_module(
        self, m: torch.nn.Module, forward: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        curr_dir = self.mod_dir
        self.mod_dir = "self." + self.path_of_module(m)
        rst = super().call_module(m, forward, args, kwargs)
        self.mod_dir = curr_dir
        return rst

    def proxy(self, node: Node) -> "ColoProxy":
        return ColoProxy(node, self)

    def create_proxy(
        self,
        kind: str,
        target: Target,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
        proxy_factory_fn: Callable[[Node], "Proxy"] = None,
    ):
        proxy: ColoProxy = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)
        unwrap_fn = lambda p: p.meta_data if isinstance(p, ColoProxy) else p
        if kind == "placeholder":
            proxy.meta_data = (
                self.meta_args[target]
                if target in self.meta_args
                else self.concrete_args.get(_truncate_suffix(target), None)
            )
        elif kind == "get_attr":
            self.disable_module_getattr = True
            try:
                attr_itr = self.root
                atoms = target.split(".")
                for atom in atoms:
                    attr_itr = getattr(attr_itr, atom)
                proxy.meta_data = attr_itr
            finally:
                self.disable_module_getattr = False
        elif kind == "call_function":
            proxy.meta_data = target(*tree_map(unwrap_fn, args), **tree_map(unwrap_fn, kwargs))
        elif kind == "call_method":
            self.disable_module_getattr = True
            try:
                if target == "__call__":
                    proxy.meta_data = unwrap_fn(args[0])(*tree_map(unwrap_fn, args[1:]), **tree_map(unwrap_fn, kwargs))
                else:
                    if target not in _TensorPropertyMethod:
                        proxy._meta_data = getattr(unwrap_fn(args[0]), target)(
                            *tree_map(unwrap_fn, args[1:]), **tree_map(unwrap_fn, kwargs)
                        )
            finally:
                self.disable_module_getattr = False
        elif kind == "call_module":
            mod = self.root.get_submodule(target)
            self.disable_module_getattr = True
            try:
                args = tree_map(unwrap_fn, args)
                kwargs = tree_map(unwrap_fn, kwargs)
                if type(mod) in self._custom_leaf_module:
                    target = self._custom_leaf_module_impl[type(mod)]
                    proxy.meta_data = target(mod, *args, **kwargs)
                else:
                    proxy.meta_data = mod.forward(*args, **kwargs)
            finally:
                self.disable_module_getattr = False
        return proxy

    def create_node(self, *args, **kwargs) -> Node:
        node = super().create_node(*args, **kwargs)
        n_info = MetaInfo(node, mod_dir=self.mod_dir, activation_checkpoint=tuple(self.ckpt_regions))
        return node

    def trace(
        self,
        root: torch.nn.Module,
        concrete_args: Optional[Dict[str, torch.Tensor]] = None,
        meta_args: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Graph:
        if meta_args is None:
            meta_args = {}

        if concrete_args is None:
            concrete_args = {}

        # check concrete and meta args have valid names
        sig = inspect.signature(root.forward)
        sig_names = set(sig.parameters.keys())
        meta_arg_names = set(meta_args.keys())
        concrete_arg_names = set(concrete_args.keys())
        non_concrete_arg_names = sig_names - concrete_arg_names
        # update concrete args with default values
        for k, v in sig.parameters.items():
            if k in sig_names - meta_arg_names and k not in concrete_args and v.default is not inspect.Parameter.empty:
                concrete_args[k] = v.default

        def _check_arg_name_valid(names: Iterable[str]):
            for name in names:
                if name not in sig_names:
                    raise ValueError(f"Argument {name} is not in the signature of {root.__class__.__name__}.forward")

        _check_arg_name_valid(meta_arg_names)
        _check_arg_name_valid(concrete_arg_names)

        self.concrete_args = concrete_args
        self.meta_args = meta_args

        with self._torch_factory_override(), self._tracer_override(), torch.no_grad():
            self.mod_dir = "self"
            self.graph = super().trace(root, concrete_args=concrete_args)
            self.mod_dir = ""
        self.graph.lint()

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

    @contextmanager
    def _tracer_override(self):
        # override the tracer to support custom modules and checkpointing
        if self.trace_act_ckpt:
            orig_ckpt_func_apply = torch.utils.checkpoint.CheckpointFunction.apply
            orig_ckpt_func_without_reentrant = torch.utils.checkpoint._checkpoint_without_reentrant_generator

            def checkpoint(run_function, preserve_rng_state=False, *args):
                self.ckpt_regions.append(self.ckpt_idx)
                out = run_function(*args)
                self.ckpt_idx = self.ckpt_regions.pop(-1) + 1
                return out

            # override the checkpoint function
            torch.utils.checkpoint.CheckpointFunction.apply = checkpoint
            torch.utils.checkpoint._checkpoint_without_reentrant = checkpoint

        # override the custom functions
        ColoProxy._func_dispatch.update({k: v for k, v in self._custom_impl.items()})

        # override the bias addition functions
        if self.bias_addition_split:
            ColoProxy._func_dispatch.update({k: v for k, v in self._bias_addition_impl.items()})

        yield

        if self.trace_act_ckpt:
            # recover the checkpoint function upon exit
            torch.utils.checkpoint.CheckpointFunction.apply = orig_ckpt_func_apply
            torch.utils.checkpoint._checkpoint_reentrant = orig_ckpt_func_without_reentrant

        ColoProxy._func_dispatch = {}

    @contextmanager
    def _torch_factory_override(self):
        # override the torch factory functions to create a proxy when the method
        # is called during ``symbolic_trace()``.
        def wrap_factory_method(target):
            @functools.wraps(target)
            def wrapper(*args, **kwargs):
                is_proxy = any(isinstance(p, ColoProxy) for p in args) | any(
                    isinstance(p, ColoProxy) for p in kwargs.values()
                )
                if is_proxy:
                    # if the arg is a proxy, then need to record this function called on this proxy
                    # e.g. torch.ones(size) where size is an input proxy
                    self.disable_module_getattr = True
                    try:
                        proxy = self.create_proxy("call_function", target, args, kwargs)
                    finally:
                        self.disable_module_getattr = False
                    return proxy
                else:
                    return target(*args, **kwargs)

            return wrapper, target

        overrides = {
            target: wrap_factory_method(getattr(torch, target))
            for target in _TorchFactoryMethod
            if callable(getattr(torch, target))
        }
        for name, (wrapper, orig) in overrides.items():
            setattr(torch, name, wrapper)

        yield

        # recover the torch factory functions upon exit
        for name, (wrapper, orig) in overrides.items():
            setattr(torch, name, orig)

    def _post_check(self, non_concrete_arg_names: Set[str]):
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

            if node.op == "output":
                node.type = None
            self.graph.lint()

    def getattr(self, attr, attr_val, parameter_proxy_cache):
        return self._module_getattr(attr, attr_val, parameter_proxy_cache)

    def _module_getattr(self, attr, attr_val, parameter_proxy_cache):
        if getattr(self, "disable_module_getattr", False):
            return attr_val

        def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
            for n, p in collection_to_search:
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        kwargs = {}
                        if "proxy_factory_fn" in inspect.signature(self.create_proxy).parameters:
                            kwargs["proxy_factory_fn"] = (
                                None
                                if not self.param_shapes_constant
                                else lambda node: ColoProxy(self, node, n, attr_val)
                            )
                        val_proxy = self.create_proxy("get_attr", n, (), {}, **kwargs)  # type: ignore[arg-type]
                        parameter_proxy_cache[n] = val_proxy
                    return parameter_proxy_cache[n]
            return None

        if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
            maybe_buffer_proxy = maybe_get_proxy_for_attr(attr_val, self.root.named_buffers(), parameter_proxy_cache)
            if maybe_buffer_proxy is not None:
                return maybe_buffer_proxy

        if isinstance(attr_val, torch.nn.Parameter):
            maybe_parameter_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_parameters(), parameter_proxy_cache
            )
            if maybe_parameter_proxy is not None:
                return maybe_parameter_proxy

        return attr_val
