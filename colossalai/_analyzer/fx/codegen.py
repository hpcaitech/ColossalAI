from typing import Any, Dict, List, Tuple

import torch

try:
    from torch.fx.graph import CodeGen
except:
    pass
from torch.fx.graph import (
    PythonCode,
    _custom_builtins,
    _format_target,
    _is_from_torch,
    _Namespace,
    _origin_type_map,
    _register_custom_builtin,
    inplace_methods,
    magic_methods,
)
from torch.fx.node import Argument, Node, _get_qualified_name, _type_repr, map_arg

import colossalai
from colossalai.fx._compatibility import compatibility

_register_custom_builtin("colossalai", "import colossalai", colossalai)


def _gen_ckpt_fn_def(label, free_vars: List[str]) -> str:
    """
    Generate the checkpoint function definition
    """
    return f"def checkpoint_{label}({', '.join(['self'] + free_vars)}):"


def _gen_ckpt_output(output_vars: List[str]) -> str:
    """
    Generate the return statement for checkpoint region
    """
    return f"return {', '.join(output_vars)}"


def _gen_ckpt_usage(label, input_vars, output_vars, use_reentrant=True):
    """
    Generate the checkpoint function call code text
    """
    outputs = ", ".join(output_vars)
    inputs = ", ".join(input_vars)
    return f"{outputs} = torch.utils.checkpoint.checkpoint(self.checkpoint_{label}, {inputs}, use_reentrant={use_reentrant})"


def _end_of_ckpt(node: Node, ckpt_level: int) -> bool:
    """
    Check if the node could end the ckpt region at `ckpt_level`
    """
    if len(node.meta["info"].activation_checkpoint) > ckpt_level:
        return node.meta["info"].activation_checkpoint[ckpt_level] is not None
    return True


def _find_input_and_output_nodes(nodes: List[Node]):
    """
    Find the input and output node names which are not found in the given list of nodes.
    """
    input_nodes = []
    output_nodes = []

    # if a node has an input node which is not in the node list
    # we treat that input node as the input of the checkpoint function
    for node in nodes:
        for input_node in node._input_nodes.keys():
            node_repr = repr(input_node)
            if input_node not in nodes and node_repr not in input_nodes:
                input_nodes.append(node_repr)

    # if a node has a user node which is not in the node list
    # we treat that user node as the node receiving the current node output
    for node in nodes:
        for output_node in node.users.keys():
            node_repr = repr(node)
            if output_node not in nodes and node_repr not in output_nodes:
                output_nodes.append(node_repr)

    return input_nodes, output_nodes


def _find_nested_ckpt_regions(node_list: List[Node], ckpt_level: int = 0):
    """
    Find the nested checkpoint regions given a list of consecutive nodes. The outputs
    will be list of tuples, each tuple is in the form of (start_index, end_index).
    """
    ckpt_regions = []
    start = -1
    end = -1
    current_region = None

    for idx, node in enumerate(node_list):
        if len(node.meta["info"].activation_checkpoint) > ckpt_level:
            act_ckpt_label = node.meta["info"].activation_checkpoint[ckpt_level]

            # this activation checkpoint label is not set yet
            # meaning this is the first node of the activation ckpt region
            if current_region is None:
                current_region = act_ckpt_label
                start = idx

            # if activation checkpoint has changed
            # we restart the tracking
            # e.g. node ckpt states = [ckpt1, ckpt2, ckpt2, ckpt2]
            if act_ckpt_label != current_region:
                assert start != -1
                ckpt_regions.append((start, idx - 1))
                current_region = act_ckpt_label
                start = idx
                end = -1

        elif current_region is not None and _end_of_ckpt(node, ckpt_level):
            # used to check the case below
            # node ckpt states = [ckpt, ckpt, non-ckpt]
            end = idx - 1
            assert start != -1 and end != -1
            ckpt_regions.append((start, end))
            start = end = -1
            current_region = None

        else:
            pass

    if current_region is not None:
        end = len(node_list) - 1
        ckpt_regions.append((start, end))
    return ckpt_regions


def emit_ckpt_func(
    body, ckpt_func, node_list: List[Node], emit_node_func, delete_unused_value_func, ckpt_level=0, in_ckpt=False
):
    """Emit ckpt function in nested way

    Args:
        body: forward code - in recursive calls, this part will be checkpoint
        functions code
        ckpt_func: checkpoint functions code - in recursive calls, this part
        will be a buffer
        node_list (List[Node]): list of torch.fx.Node
        emit_node_func: function to emit a node
        delete_unused_value_func: function to delete unused value
        level (int, optional): checkpoint level. Defaults to 0.
        in_ckpt (bool, optional): indicates wether the func is in recursive
        call. Defaults to False.
    """
    inputs, outputs = _find_input_and_output_nodes(node_list)

    # label given by each layer, e.g. if you are currently at level (0, 1, 1)
    # the label will be '0_1_1'
    label = "_".join([str(idx) for idx in node_list[0].meta["info"].activation_checkpoint[: ckpt_level + 1]])
    ckpt_fn_def = _gen_ckpt_fn_def(label, inputs)
    ckpt_func.append(f"{ckpt_fn_def}\n")

    # if there is more level to fetch
    if ckpt_level + 1 < max(map(lambda node: len(node.meta["info"].activation_checkpoint), node_list)):
        ckpt_regions = _find_nested_ckpt_regions(node_list, ckpt_level + 1)
        start_idx = [item[0] for item in ckpt_regions]
        end_idx = [item[1] for item in ckpt_regions]

        # use ckpt_func_buffer to store nested checkpoint functions
        ckpt_func_buffer = []
        node_idx = 0
        while 1:
            if node_idx >= len(node_list):
                break

            if node_idx in start_idx:
                ckpt_node_list = node_list[node_idx : end_idx[start_idx.index(node_idx)] + 1]
                emit_ckpt_func(
                    ckpt_func,
                    ckpt_func_buffer,
                    ckpt_node_list,
                    emit_node_func,
                    delete_unused_value_func,
                    ckpt_level + 1,
                    True,
                )
                node_idx += len(ckpt_node_list)

            else:
                node = node_list[node_idx]
                emit_node_func(node, ckpt_func)
                ckpt_func[-1] = "    " + ckpt_func[-1]
                delete_unused_value_func(node, ckpt_func)
                node_idx += 1

        ckpt_func.append("    " + _gen_ckpt_output(outputs) + "\n\n")
        ckpt_func += ckpt_func_buffer

    # last level
    else:
        for node in node_list:
            emit_node_func(node, ckpt_func)
            ckpt_func[-1] = "    " + ckpt_func[-1]
            delete_unused_value_func(node, ckpt_func)

        ckpt_func.append("    " + _gen_ckpt_output(outputs) + "\n\n")

    usage = _gen_ckpt_usage(label, inputs, outputs, False) + "\n"
    if in_ckpt:
        usage = "    " + usage
    body.append(usage)


def emit_code_with_activation_checkpoint(body, ckpt_func, nodes, emit_node_func, delete_unused_value_func):
    """Emit code with nested activation checkpoint
    When we detect some of the annotation is a , we will use
    this function to emit the activation checkpoint codes.

    Args:
        body: forward code
        ckpt_func: checkpoint functions code
        nodes: graph.nodes
        emit_node_func: function to emit node
        delete_unused_value_func: function to remove the unused value
    """
    ckpt_regions = _find_nested_ckpt_regions(nodes, 0)
    start_idx = [item[0] for item in ckpt_regions]
    end_idx = [item[1] for item in ckpt_regions]
    node_list = list(nodes)

    node_idx = 0
    while 1:
        # break if we finish the processing all the nodes
        if node_idx >= len(node_list):
            break

        # process ckpt_regions
        if node_idx in start_idx:
            ckpt_node_list = node_list[node_idx : end_idx[start_idx.index(node_idx)] + 1]
            emit_ckpt_func(body, ckpt_func, ckpt_node_list, emit_node_func, delete_unused_value_func)
            node_idx += len(ckpt_node_list)

        # process node in forward function
        else:
            node = node_list[node_idx]
            emit_node_func(node, body)
            delete_unused_value_func(node, body)
            node_idx += 1


@compatibility(is_backward_compatible=True)
class ActivationCheckpointCodeGen(CodeGen):
    def _gen_python_code(self, nodes, root_module: str, namespace: _Namespace, verbose=None) -> PythonCode:
        free_vars: List[str] = []
        body: List[str] = []
        globals_: Dict[str, Any] = {}
        wrapped_fns: Dict[str, None] = {}

        # Wrap string in list to pass by reference
        maybe_return_annotation: List[str] = [""]

        def add_global(name_hint: str, obj: Any):
            """Add an obj to be tracked as a global.
            We call this for names that reference objects external to the
            Graph, like functions or types.
            Returns: the global name that should be used to reference 'obj' in generated source.
            """
            if _is_from_torch(obj) and obj != torch.device:  # to support registering torch.device
                # HACK: workaround for how torch custom ops are registered. We
                # can't import them like normal modules so they must retain their
                # fully qualified name.
                return _get_qualified_name(obj)

            # normalize the name hint to get a proper identifier
            global_name = namespace.create_name(name_hint, obj)

            if global_name in globals_:
                assert globals_[global_name] is obj
                return global_name
            globals_[global_name] = obj
            return global_name

        # Pre-fill the globals table with registered builtins.
        for name, (_, obj) in _custom_builtins.items():
            add_global(name, obj)

        def type_repr(o: Any):
            if o == ():
                # Empty tuple is used for empty tuple type annotation Tuple[()]
                return "()"

            typename = _type_repr(o)

            if hasattr(o, "__origin__"):
                # This is a generic type, e.g. typing.List[torch.Tensor]
                origin_type = _origin_type_map.get(o.__origin__, o.__origin__)
                origin_typename = add_global(_type_repr(origin_type), origin_type)

                if hasattr(o, "__args__"):
                    # Assign global names for each of the inner type variables.
                    args = [type_repr(arg) for arg in o.__args__]

                    if len(args) == 0:
                        # Bare type, such as `typing.Tuple` with no subscript
                        # This code-path used in Python < 3.9
                        return origin_typename

                    return f'{origin_typename}[{",".join(args)}]'
                else:
                    # Bare type, such as `typing.Tuple` with no subscript
                    # This code-path used in Python 3.9+
                    return origin_typename

            # Common case: this is a regular module name like 'foo.bar.baz'
            return add_global(typename, o)

        def _format_args(args: Tuple[Argument, ...], kwargs: Dict[str, Argument]) -> str:
            def _get_repr(arg):
                # Handle NamedTuples (if it has `_fields`) via add_global.
                if isinstance(arg, tuple) and hasattr(arg, "_fields"):
                    qualified_name = _get_qualified_name(type(arg))
                    global_name = add_global(qualified_name, type(arg))
                    return f"{global_name}{repr(tuple(arg))}"
                return repr(arg)

            args_s = ", ".join(_get_repr(a) for a in args)
            kwargs_s = ", ".join(f"{k} = {_get_repr(v)}" for k, v in kwargs.items())
            if args_s and kwargs_s:
                return f"{args_s}, {kwargs_s}"
            return args_s or kwargs_s

        # Run through reverse nodes and record the first instance of a use
        # of a given node. This represents the *last* use of the node in the
        # execution order of the program, which we will use to free unused
        # values
        node_to_last_use: Dict[Node, Node] = {}
        user_to_last_uses: Dict[Node, List[Node]] = {}

        def register_last_uses(n: Node, user: Node):
            if n not in node_to_last_use:
                node_to_last_use[n] = user
                user_to_last_uses.setdefault(user, []).append(n)

        for node in reversed(nodes):
            map_arg(node.args, lambda n: register_last_uses(n, node))
            map_arg(node.kwargs, lambda n: register_last_uses(n, node))

        # NOTE: we add a variable to distinguish body and ckpt_func
        def delete_unused_values(user: Node, body):
            """
            Delete values after their last use. This ensures that values that are
            not used in the remainder of the code are freed and the memory usage
            of the code is optimal.
            """
            if user.op == "placeholder":
                return
            if user.op == "output":
                body.append("\n")
                return
            nodes_to_delete = user_to_last_uses.get(user, [])
            if len(nodes_to_delete):
                to_delete_str = " = ".join([repr(n) for n in nodes_to_delete] + ["None"])
                body.append(f";  {to_delete_str}\n")
            else:
                body.append("\n")

        # NOTE: we add a variable to distinguish body and ckpt_func
        def emit_node(node: Node, body):
            maybe_type_annotation = "" if node.type is None else f" : {type_repr(node.type)}"
            if node.op == "placeholder":
                assert isinstance(node.target, str)
                maybe_default_arg = "" if not node.args else f" = {repr(node.args[0])}"
                free_vars.append(f"{node.target}{maybe_type_annotation}{maybe_default_arg}")
                raw_name = node.target.replace("*", "")
                if raw_name != repr(node):
                    body.append(f"{repr(node)} = {raw_name}\n")
                return
            elif node.op == "call_method":
                assert isinstance(node.target, str)
                body.append(
                    f"{repr(node)}{maybe_type_annotation} = {_format_target(repr(node.args[0]), node.target)}"
                    f"({_format_args(node.args[1:], node.kwargs)})"
                )
                return
            elif node.op == "call_function":
                assert callable(node.target)
                # pretty print operators
                if node.target.__module__ == "_operator" and node.target.__name__ in magic_methods:
                    assert isinstance(node.args, tuple)
                    body.append(
                        f"{repr(node)}{maybe_type_annotation} = "
                        f"{magic_methods[node.target.__name__].format(*(repr(a) for a in node.args))}"
                    )
                    return

                # pretty print inplace operators; required for jit.script to work properly
                # not currently supported in normal FX graphs, but generated by torchdynamo
                if node.target.__module__ == "_operator" and node.target.__name__ in inplace_methods:
                    body.append(
                        f"{inplace_methods[node.target.__name__].format(*(repr(a) for a in node.args))};  "
                        f"{repr(node)}{maybe_type_annotation} = {repr(node.args[0])}"
                    )
                    return

                qualified_name = _get_qualified_name(node.target)
                global_name = add_global(qualified_name, node.target)
                # special case for getattr: node.args could be 2-argument or 3-argument
                # 2-argument: attribute access; 3-argument: fall through to attrib function call with default value
                if (
                    global_name == "getattr"
                    and isinstance(node.args, tuple)
                    and isinstance(node.args[1], str)
                    and node.args[1].isidentifier()
                    and len(node.args) == 2
                ):
                    body.append(
                        f"{repr(node)}{maybe_type_annotation} = {_format_target(repr(node.args[0]), node.args[1])}"
                    )
                    return
                body.append(
                    f"{repr(node)}{maybe_type_annotation} = {global_name}({_format_args(node.args, node.kwargs)})"
                )
                if node.meta.get("is_wrapped", False):
                    wrapped_fns.setdefault(global_name)
                return
            elif node.op == "call_module":
                assert isinstance(node.target, str)
                body.append(
                    f"{repr(node)}{maybe_type_annotation} = "
                    f"{_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})"
                )
                return
            elif node.op == "get_attr":
                assert isinstance(node.target, str)
                body.append(f"{repr(node)}{maybe_type_annotation} = {_format_target(root_module, node.target)}")
                return
            elif node.op == "output":
                if node.type is not None:
                    maybe_return_annotation[0] = f" -> {type_repr(node.type)}"
                body.append(self.generate_output(node.args[0]))
                return
            raise NotImplementedError(f"node: {node.op} {node.target}")

        # Modified for activation checkpointing
        ckpt_func = []
        emit_code_with_activation_checkpoint(body, ckpt_func, nodes, emit_node, delete_unused_values)

        if len(body) == 0:
            # If the Graph has no non-placeholder nodes, no lines for the body
            # have been emitted. To continue to have valid Python code, emit a
            # single pass statement
            body.append("pass\n")

        if len(wrapped_fns) > 0:
            wrap_name = add_global("wrap", torch.fx.wrap)
            wrap_stmts = "\n".join([f'{wrap_name}("{name}")' for name in wrapped_fns])
        else:
            wrap_stmts = ""

        if self._body_transformer:
            body = self._body_transformer(body)

        for name, value in self.additional_globals():
            add_global(name, value)

        prologue = self.gen_fn_def(free_vars, maybe_return_annotation[0])
        prologue = "".join(ckpt_func) + prologue
        prologue = prologue

        code = "".join(body)
        code = "\n".join("    " + line for line in code.split("\n"))
        fn_code = f"""
{wrap_stmts}
{prologue}
{code}"""
        return PythonCode(fn_code, globals_, {})
