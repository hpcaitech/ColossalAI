from typing import Any, Callable, Dict, Iterable, List, Tuple

import torch

import colossalai
from colossalai.fx._compatibility import is_compatible_with_meta
from colossalai.fx.codegen.activation_checkpoint_codegen import CODEGEN_AVAILABLE

AUTOCHUNK_AVAILABLE = CODEGEN_AVAILABLE and is_compatible_with_meta()

if AUTOCHUNK_AVAILABLE:
    from torch.fx.graph import (
        CodeGen,
        PythonCode,
        _custom_builtins,
        _CustomBuiltin,
        _format_target,
        _is_from_torch,
        _Namespace,
        _origin_type_map,
        inplace_methods,
        magic_methods,
    )

from torch.fx.node import Argument, Node, _get_qualified_name, _type_repr, map_arg

from .search_chunk import SearchChunk
from .utils import delete_free_var_from_last_use, get_logger, get_node_name, get_node_shape


def _gen_chunk_slice_dim(chunk_dim: int, chunk_indice_name: str, shape: List) -> str:
    """
    Generate chunk slice string, eg. [:, :, chunk_idx_name:chunk_idx_name + chunk_size, :]

    Args:
        chunk_dim (int)
        chunk_indice_name (str): chunk indice name
        shape (List): node shape

    Returns:
        new_shape (str): return slice
    """
    new_shape = "["
    for idx, _ in enumerate(shape):
        if idx == chunk_dim:
            new_shape += "%s:%s + chunk_size" % (chunk_indice_name, chunk_indice_name)
        else:
            new_shape += ":"
        new_shape += ", "
    new_shape = new_shape[:-2] + "]"
    return new_shape


def _gen_loop_start(chunk_input: List[Node], chunk_output: List[Node], chunk_output_dim: int, chunk_size=2) -> str:
    """
    Generate chunk loop start

    eg. chunk_result = torch.empty([100, 100], dtype=input_node.dtype, device=input_node.device)
        chunk_size = 32
        for chunk_idx in range(0, 100, 32):
            ......

    Args:
        chunk_input (List[Node]): chunk input node
        chunk_output (Node): chunk output node
        chunk_output_dim (int): chunk output node chunk dim
        chunk_size (int): chunk size. Defaults to 2.

    Returns:
        context (str): generated str
    """
    input_node = chunk_input[0]

    context = ""
    for i in range(len(chunk_output)):
        shape_str = str(list(get_node_shape(chunk_output[i])))
        if get_node_name(chunk_output[i]) in ["split", "unbind"]:
            tensor_str = "torch.empty(%s, dtype=%s.dtype, device=%s.device), " % (
                shape_str,
                input_node.name,
                input_node.name,
            )
            tensor_str = tensor_str * len(chunk_output[i].meta["tensor_meta"])
            tensor_str = "[" + tensor_str[:-2] + "]"
            context += "%s = %s;  " % (chunk_output[i].name, tensor_str)
        else:
            context += "%s = torch.empty(%s, dtype=%s.dtype, device=%s.device);  " % (
                chunk_output[i].name,
                shape_str,
                input_node.name,
                input_node.name,
            )

    out_shape = get_node_shape(chunk_output[0])
    chunk_shape = out_shape[chunk_output_dim[0]]
    context += "chunk_size = %d\nfor chunk_idx in range(0, %d, chunk_size):\n" % (chunk_size, chunk_shape)
    return context


def _gen_loop_end(
    chunk_inputs: List[Node],
    chunk_non_compute_inputs: List[Node],
    node_list: List[Node],
    chunk_outputs_idx: int,
    chunk_outputs_non_tensor: List[Node],
    search_chunk: SearchChunk,
) -> str:
    """
    Generate chunk loop end

    eg.     chunk_result[chunk_idx:chunk_idx + chunk_size] = output_node
        output_node = chunk_result; xx = None; xx = None

    Args:
        chunk_inputs (List[Node]): chunk input node
        chunk_non_compute_inputs (List[Node]): input node without chunk
        chunk_outputs (Node): chunk output node
        chunk_outputs_dim (int): chunk output node chunk dim
        node_list (List)

    Returns:
        context (str): generated str
    """
    context = "chunk_size = None"
    # determine if its the last use for chunk input
    for chunk_input in chunk_inputs + chunk_non_compute_inputs:
        if all([search_chunk.node_mgr.find_node_idx(user) <= chunk_outputs_idx for user in chunk_input.users.keys()]):
            context += ";  %s = None" % chunk_input.name
    for chunk_output_non_tensor, chunk_output_non_tensor_val in chunk_outputs_non_tensor.items():
        context += ";  %s = %s" % (chunk_output_non_tensor.name, chunk_output_non_tensor_val)
    context += "\n"
    return context


def _replace_name(context: str, name_from: str, name_to: str) -> str:
    """
    replace node name
    """
    patterns = [(" ", " "), (" ", "."), (" ", ","), ("(", ")"), ("(", ","), (" ", ")"), (" ", ""), ("", " ")]
    for p in patterns:
        source = p[0] + name_from + p[1]
        target = p[0] + name_to + p[1]
        if source in context:
            context = context.replace(source, target)
            break
    return context


def _replace_reshape_size(context: str, node_name: str, reshape_size_dict: Dict) -> str:
    """
    replace reshape size, some may have changed due to chunk
    """
    if node_name not in reshape_size_dict:
        return context
    context = context.replace(reshape_size_dict[node_name][0], reshape_size_dict[node_name][1])
    return context


def _replace_new_tensor_like_shape(
    search_chunk: SearchChunk,
    chunk_infos: List[Dict],
    region_idx: int,
    node_idx: int,
    node: Node,
    body: List[str],
) -> List[str]:
    """
    add chunk slice for new tensor op such as ones like
    """
    if get_node_name(node) in ["ones_like", "zeros_like", "empty_like"]:
        meta_node = search_chunk.node_mgr.get_node_by_idx(node_idx)
        chunk_dim = chunk_infos[region_idx]["node_chunk_dim"][meta_node]["chunk_dim"]
        if get_node_shape(meta_node)[chunk_dim] != 1:
            source_node = meta_node.args[0].args[0]
            if (
                source_node not in chunk_infos[region_idx]["node_chunk_dim"]
                or chunk_infos[region_idx]["node_chunk_dim"][source_node]["chunk_dim"] is None
            ):
                chunk_slice = _gen_chunk_slice_dim(chunk_dim, "chunk_idx", get_node_shape(node))
                body[-1] = _replace_name(body[-1], node.args[0].name, node.args[0].name + chunk_slice)
    return body


def _replace_new_tensor_shape(
    search_chunk: SearchChunk,
    chunk_infos: List[Dict],
    region_idx: int,
    node_idx: int,
    node: Node,
    body: List[str],
) -> List[str]:
    """
    add chunk slice for new tensor op such as ones
    """
    if get_node_name(node) in ["ones", "zeros", "empty"]:
        meta_node = search_chunk.node_mgr.get_node_by_idx(node_idx)
        chunk_dim = chunk_infos[region_idx]["node_chunk_dim"][meta_node]["chunk_dim"]
        if chunk_dim is None:
            return
        if get_node_shape(meta_node)[chunk_dim] == 1:
            return
        origin_shape = str(node.args)
        new_shape = list(node.args)
        new_shape[chunk_dim] = "min(chunk_size, %d - chunk_idx)" % get_node_shape(meta_node)[chunk_dim]
        new_shape = str(new_shape)
        new_shape = new_shape.replace("'", "")
        body[-1] = _replace_name(body[-1], origin_shape[1:-1], new_shape[1:-1])
    return body


def _add_node_slice(
    chunk_nodes: List[Node],
    region_idx: int,
    chunk_nodes_dim: Dict,
    node_idx: int,
    body: List[str],
    node: Node,
) -> List[str]:
    """
    add chunk slice for input nodes
    """
    for chunk_node_idx, chunk_node in enumerate(chunk_nodes[region_idx]):
        # inputs node
        if isinstance(chunk_nodes_dim[region_idx][chunk_node_idx], dict):
            for idx, dim in chunk_nodes_dim[region_idx][chunk_node_idx].items():
                if idx == node_idx:
                    chunk_slice = _gen_chunk_slice_dim(dim[0], "chunk_idx", get_node_shape(chunk_node))
                    body[-1] = _replace_name(body[-1], chunk_node.name, chunk_node.name + chunk_slice)
        # outputs node
        else:
            if chunk_node.name == node.name or (chunk_node.name in [i.name for i in node.all_input_nodes]):
                chunk_slice = _gen_chunk_slice_dim(
                    chunk_nodes_dim[region_idx][chunk_node_idx], "chunk_idx", get_node_shape(chunk_node)
                )
                if get_node_name(chunk_node) in ["split", "unbind"]:
                    split_chunk_slice = ""
                    for i in range(len(chunk_node.meta["tensor_meta"])):
                        split_chunk_slice += "%s[%d]%s, " % (chunk_node.name, i, chunk_slice)
                    split_chunk_slice = split_chunk_slice[:-2]
                    body[-1] = _replace_name(body[-1], chunk_node.name, split_chunk_slice)
                else:
                    body[-1] = _replace_name(body[-1], chunk_node.name, chunk_node.name + chunk_slice)
    return body


def emit_code_with_chunk(
    body: List[str],
    nodes: Iterable[Node],
    emit_node_func: Callable,
    delete_unused_value_func: Callable,
    search_chunk: SearchChunk,
    chunk_infos: List,
    eval_mem: bool = False,
):
    """
    Emit code with chunk according to chunk_infos.

    It will generate a for loop in chunk regions, and
    replace inputs and outputs of regions with chunked variables.

    Args:
        body: forward code
        nodes: graph.nodes
        emit_node_func: function to emit node
        delete_unused_value_func: function to remove the unused value
        search_chunk: the class to search all chunks
        chunk_infos: store all information about all chunks.
    """
    node_list = list(nodes)

    # chunk region
    chunk_starts = [i["region"][0] for i in chunk_infos]
    chunk_ends = [i["region"][1] for i in chunk_infos]

    # chunk inputs
    chunk_inputs = [i["inputs"] for i in chunk_infos]  # input with chunk
    chunk_inputs_non_chunk = [i["inputs_non_chunk"] for i in chunk_infos]  # input without chunk
    chunk_inputs_dim = [i["inputs_dim"] for i in chunk_infos]  # input chunk dim
    chunk_inputs_names = [j.name for i in chunk_inputs for j in i] + [j.name for i in chunk_inputs_non_chunk for j in i]

    # chunk outputs
    chunk_outputs = [i["outputs"] for i in chunk_infos]
    chunk_outputs_non_tensor = [i["outputs_non_tensor"] for i in chunk_infos]
    chunk_outputs_dim = [i["outputs_dim"] for i in chunk_infos]

    node_list = search_chunk.reorder_graph.reorder_node_list(node_list)
    node_idx = 0
    region_idx = 0
    within_chunk_region = False

    if eval_mem:
        body.append("init_memory = torch.cuda.memory_allocated() / 1024**2\n")

    while node_idx < len(node_list):
        node = node_list[node_idx]

        # if is chunk start, generate for loop start
        if node_idx in chunk_starts:
            within_chunk_region = True
            region_idx = chunk_starts.index(node_idx)
            body.append(
                _gen_loop_start(
                    chunk_inputs[region_idx],
                    chunk_outputs[region_idx],
                    chunk_outputs_dim[region_idx],
                    chunk_infos[region_idx]["chunk_size"],
                )
            )

        if within_chunk_region:
            emit_node_func(node, body)
            # replace input var with chunk var
            body = _add_node_slice(chunk_inputs, region_idx, chunk_inputs_dim, node_idx, body, node)
            # replace output var with chunk var
            body = _add_node_slice(chunk_outputs, region_idx, chunk_outputs_dim, node_idx, body, node)
            # new tensor like
            body = _replace_new_tensor_like_shape(search_chunk, chunk_infos, region_idx, node_idx, node, body)
            # new tensor
            body = _replace_new_tensor_shape(search_chunk, chunk_infos, region_idx, node_idx, node, body)
            # reassign reshape size
            body[-1] = _replace_reshape_size(body[-1], node.name, chunk_infos[region_idx]["reshape_size"])
            body[-1] = "    " + body[-1]
            delete_unused_value_func(node, body, chunk_inputs_names)
            if eval_mem:
                body.append(
                    "    if chunk_idx == 0:\n        print('%s', torch.cuda.max_memory_allocated() / 1024**2 - init_memory);  torch.cuda.reset_peak_memory_stats()\n"
                    % (node.name)
                )
        else:
            emit_node_func(node, body)
            if node_idx not in chunk_inputs:
                delete_unused_value_func(node, body, chunk_inputs_names)
            if eval_mem:
                body.append(
                    "print('%s', torch.cuda.max_memory_allocated() / 1024**2 - init_memory);  torch.cuda.reset_peak_memory_stats()\n"
                    % (node.name)
                )

        # generate chunk region end
        if node_idx in chunk_ends:
            body.append(
                _gen_loop_end(
                    chunk_inputs[region_idx],
                    chunk_inputs_non_chunk[region_idx],
                    node_list,
                    chunk_ends[region_idx],
                    chunk_outputs_non_tensor[region_idx],
                    search_chunk,
                )
            )
            within_chunk_region = False

        node_idx += 1


if AUTOCHUNK_AVAILABLE:

    class AutoChunkCodeGen(CodeGen):
        def __init__(
            self,
            meta_graph,
            max_memory: int = None,
            print_mem: bool = False,
            print_progress: bool = False,
            eval_mem: bool = False,
        ) -> None:
            super().__init__()
            self.eval_mem = eval_mem
            # find the chunk regions
            self.search_chunk = SearchChunk(meta_graph, max_memory, print_mem, print_progress)
            self.chunk_infos = self.search_chunk.search_region()
            if print_progress:
                get_logger().info("AutoChunk start codegen")

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

            # set _custom_builtins here so that we needn't import colossalai in forward
            _custom_builtins["colossalai"] = _CustomBuiltin("import colossalai", colossalai)

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

            delete_free_var_from_last_use(user_to_last_uses)

            # NOTE: we add a variable to distinguish body and ckpt_func
            def delete_unused_values(user: Node, body, to_keep=[]):
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
                nodes_to_delete = [i for i in nodes_to_delete if i.name not in to_keep]
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

            # if any node has a list of labels for activation_checkpoint, we
            # will use nested type of activation checkpoint codegen
            emit_code_with_chunk(
                body, nodes, emit_node, delete_unused_values, self.search_chunk, self.chunk_infos, self.eval_mem
            )

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

            # as we need colossalai.utils.checkpoint, we need to import colossalai
            # in forward function
            prologue = self.gen_fn_def(free_vars, maybe_return_annotation[0])
            prologue = "".join(ckpt_func) + prologue
            prologue = prologue

            code = "".join(body)
            code = "\n".join("    " + line for line in code.split("\n"))
            fn_code = f"""
{wrap_stmts}

{prologue}
{code}"""
            # print(fn_code)
            return PythonCode(fn_code, globals_)
