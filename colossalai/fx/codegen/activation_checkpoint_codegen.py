import colossalai
import torch
from typing import List, Callable, Any, Tuple, Dict

try:
    from torch.fx.node import Node, Argument, map_arg, _type_repr, _get_qualified_name
    from torch.fx.graph import _Namespace, PythonCode, _custom_builtins, _is_from_torch, _format_target, magic_methods, CodeGen, _origin_type_map, inplace_methods, _CustomBuiltin
    CODEGEN_AVAILABLE = True
except:
    from torch.fx.graph import _Namespace, PythonCode, _custom_builtins, _is_from_torch, _format_target, magic_methods, _origin_type_map, _format_args, _CustomBuiltin
    from torch.fx.node import Node, Argument, map_arg, _type_repr, _get_qualified_name
    CODEGEN_AVAILABLE = False

if CODEGEN_AVAILABLE:
    __all__ = ['ActivationCheckpointCodeGen']
else:
    __all__ = ['python_code_with_activation_checkpoint']


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


def _find_ckpt_regions(nodes: List[Node]):
    """
    Find the checkpoint regions given a list of consecutive nodes. The outputs will be list
    of tuples, each tuple is in the form of (start_index, end_index).
    """
    ckpt_nodes = []
    ckpt_regions = []
    start = -1
    end = -1
    current_region = None

    for idx, node in enumerate(nodes):
        if hasattr(node, 'activation_checkpoint'):
            act_ckpt_label = node.activation_checkpoint

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
        elif current_region is not None and not hasattr(node, 'activation_checkpoint'):
            # used to check the case below
            # node ckpt states = [ckpt, ckpt, non-ckpt]
            end = idx - 1
            assert start != -1 and end != -1
            ckpt_regions.append((start, end))
            start = end = -1
            current_region = None
        else:
            pass
    return ckpt_regions


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


def _gen_ckpt_usage(label, activation_offload, input_vars, output_vars, use_reentrant=True):
    """
    Generate the checkpoint function call code text
    """
    outputs = ', '.join(output_vars)
    inputs = ', '.join(input_vars)
    return f'{outputs} = colossalai.utils.activation_checkpoint.checkpoint(self.checkpoint_{label}, {activation_offload}, {inputs}, use_reentrant={use_reentrant})'


def emit_code_with_activation_checkpoint(body, ckpt_func, nodes, emit_node_func, delete_unused_value_func):
    # find the activation checkpoint regions
    ckpt_regions = _find_ckpt_regions(nodes)
    start_idx = [item[0] for item in ckpt_regions]
    end_idx = [item[1] for item in ckpt_regions]
    input_vars = []
    output_vars = []
    within_ckpt_region = False

    node_list = list(nodes)

    # find the input and output var names for each region
    for idx, (start, end) in enumerate(ckpt_regions):
        ckpt_node_list = node_list[start:end + 1]
        inputs, outputs = _find_input_and_output_nodes(ckpt_node_list)
        input_vars.append(inputs)
        output_vars.append(outputs)

    # append code text to body
    for idx, node in enumerate(node_list):
        # if this is the first node of the ckpt region
        # append the ckpt function defition
        if idx in start_idx:
            label = start_idx.index(idx)
            ckpt_fn_def = _gen_ckpt_fn_def(label, input_vars[label])
            ckpt_func.append(f'{ckpt_fn_def}\n')
            within_ckpt_region = True

        # NOTE: emit_node does not emit a string with newline. It depends
        # on delete_unused_values to append one
        # NOTE: currently we separate body and ckpt_func definition
        if within_ckpt_region:
            emit_node_func(node, ckpt_func)
            ckpt_func[-1] = '    ' + ckpt_func[-1]
            delete_unused_value_func(node, ckpt_func)
        else:
            emit_node_func(node, body)
            delete_unused_value_func(node, body)

        if idx in end_idx:
            # if this is the last node of the ckpt region
            # generate return statement
            label = end_idx.index(idx)
            return_statement = _gen_ckpt_output(output_vars[label])
            return_statement = f'    {return_statement}\n\n'
            ckpt_func.append(return_statement)

            # we need to check if the checkpoint need to offload the input
            start_node_idx = start_idx[label]
            if hasattr(node_list[start_node_idx], 'activation_offload'):
                activation_offload = node_list[start_node_idx].activation_offload
            else:
                activation_offload = False

            # we need to check if the checkpoint need use_reentrant=False
            use_reentrant = True
            non_leaf_input = 0
            for var in input_vars[label]:
                input_node = next(item for item in node_list if item.name == var)
                if input_node.op != "placeholder":
                    non_leaf_input = 1
                for user in input_node.users:
                    if hasattr(user, "activation_checkpoint"):
                        if user.activation_checkpoint == label:
                            if user.op == "call_module":
                                if hasattr(user.graph.owning_module.get_submodule(user.target), "inplace"):
                                    use_reentrant = not user.graph.owning_module.get_submodule(user.target).inplace

                            elif user.op == "call_function":
                                if "inplace" in user.kwargs:
                                    use_reentrant = not user.kwargs["inplace"]

            # if all the inputs are leaf nodes, we need to set use_reentrant = False
            if not non_leaf_input:
                use_reentrant = False

            # generate checkpoint function call in a new line
            usage = _gen_ckpt_usage(label, activation_offload, input_vars[label], output_vars[label], use_reentrant)
            usage += '\n'
            body.append(usage)
            within_ckpt_region = False


if CODEGEN_AVAILABLE:

    class ActivationCheckpointCodeGen(CodeGen):

        def _gen_python_code(self, nodes, root_module: str, namespace: _Namespace) -> PythonCode:
            free_vars: List[str] = []
            body: List[str] = []
            globals_: Dict[str, Any] = {}
            wrapped_fns: Dict[str, None] = {}

            # Wrap string in list to pass by reference
            maybe_return_annotation: List[str] = ['']

            def add_global(name_hint: str, obj: Any):
                """Add an obj to be tracked as a global.

                We call this for names that reference objects external to the
                Graph, like functions or types.

                Returns: the global name that should be used to reference 'obj' in generated source.
                """
                if _is_from_torch(obj) and obj != torch.device:    # to support registering torch.device
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
                    return '()'

                typename = _type_repr(o)

                if hasattr(o, '__origin__'):
                    # This is a generic type, e.g. typing.List[torch.Tensor]
                    origin_type = _origin_type_map.get(o.__origin__, o.__origin__)
                    origin_typename = add_global(_type_repr(origin_type), origin_type)

                    if hasattr(o, '__args__'):
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
                    if isinstance(arg, tuple) and hasattr(arg, '_fields'):
                        qualified_name = _get_qualified_name(type(arg))
                        global_name = add_global(qualified_name, type(arg))
                        return f"{global_name}{repr(tuple(arg))}"
                    return repr(arg)

                args_s = ', '.join(_get_repr(a) for a in args)
                kwargs_s = ', '.join(f'{k} = {_get_repr(v)}' for k, v in kwargs.items())
                if args_s and kwargs_s:
                    return f'{args_s}, {kwargs_s}'
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
                if user.op == 'placeholder':
                    return
                if user.op == 'output':
                    body.append('\n')
                    return
                nodes_to_delete = user_to_last_uses.get(user, [])
                if len(nodes_to_delete):
                    to_delete_str = ' = '.join([repr(n) for n in nodes_to_delete] + ['None'])
                    body.append(f';  {to_delete_str}\n')
                else:
                    body.append('\n')

            # NOTE: we add a variable to distinguish body and ckpt_func
            def emit_node(node: Node, body):
                maybe_type_annotation = '' if node.type is None else f' : {type_repr(node.type)}'
                if node.op == 'placeholder':
                    assert isinstance(node.target, str)
                    maybe_default_arg = '' if not node.args else f' = {repr(node.args[0])}'
                    free_vars.append(f'{node.target}{maybe_type_annotation}{maybe_default_arg}')
                    raw_name = node.target.replace('*', '')
                    if raw_name != repr(node):
                        body.append(f'{repr(node)} = {raw_name}\n')
                    return
                elif node.op == 'call_method':
                    assert isinstance(node.target, str)
                    body.append(
                        f'{repr(node)}{maybe_type_annotation} = {_format_target(repr(node.args[0]), node.target)}'
                        f'({_format_args(node.args[1:], node.kwargs)})')
                    return
                elif node.op == 'call_function':
                    assert callable(node.target)
                    # pretty print operators
                    if node.target.__module__ == '_operator' and node.target.__name__ in magic_methods:
                        assert isinstance(node.args, tuple)
                        body.append(f'{repr(node)}{maybe_type_annotation} = '
                                    f'{magic_methods[node.target.__name__].format(*(repr(a) for a in node.args))}')
                        return

                    # pretty print inplace operators; required for jit.script to work properly
                    # not currently supported in normal FX graphs, but generated by torchdynamo
                    if node.target.__module__ == '_operator' and node.target.__name__ in inplace_methods:
                        body.append(f'{inplace_methods[node.target.__name__].format(*(repr(a) for a in node.args))};  '
                                    f'{repr(node)}{maybe_type_annotation} = {repr(node.args[0])}')
                        return

                    qualified_name = _get_qualified_name(node.target)
                    global_name = add_global(qualified_name, node.target)
                    # special case for getattr: node.args could be 2-argument or 3-argument
                    # 2-argument: attribute access; 3-argument: fall through to attrib function call with default value
                    if global_name == 'getattr' and \
                    isinstance(node.args, tuple) and \
                    isinstance(node.args[1], str) and \
                    node.args[1].isidentifier() and \
                    len(node.args) == 2:
                        body.append(
                            f'{repr(node)}{maybe_type_annotation} = {_format_target(repr(node.args[0]), node.args[1])}')
                        return
                    body.append(
                        f'{repr(node)}{maybe_type_annotation} = {global_name}({_format_args(node.args, node.kwargs)})')
                    if node.meta.get('is_wrapped', False):
                        wrapped_fns.setdefault(global_name)
                    return
                elif node.op == 'call_module':
                    assert isinstance(node.target, str)
                    body.append(f'{repr(node)}{maybe_type_annotation} = '
                                f'{_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})')
                    return
                elif node.op == 'get_attr':
                    assert isinstance(node.target, str)
                    body.append(f'{repr(node)}{maybe_type_annotation} = {_format_target(root_module, node.target)}')
                    return
                elif node.op == 'output':
                    if node.type is not None:
                        maybe_return_annotation[0] = f" -> {type_repr(node.type)}"
                    body.append(self.generate_output(node.args[0]))
                    return
                raise NotImplementedError(f'node: {node.op} {node.target}')

            # Modified for activation checkpointing
            ckpt_func = []
            emit_code_with_activation_checkpoint(body, ckpt_func, nodes, emit_node, delete_unused_values)

            if len(body) == 0:
                # If the Graph has no non-placeholder nodes, no lines for the body
                # have been emitted. To continue to have valid Python code, emit a
                # single pass statement
                body.append('pass\n')

            if len(wrapped_fns) > 0:
                wrap_name = add_global('wrap', torch.fx.wrap)
                wrap_stmts = '\n'.join([f'{wrap_name}("{name}")' for name in wrapped_fns])
            else:
                wrap_stmts = ''

            if self._body_transformer:
                body = self._body_transformer(body)

            for name, value in self.additional_globals():
                add_global(name, value)

            # as we need colossalai.utils.checkpoint, we need to import colossalai
            # in forward function
            # TODO: Remove inline import
            prologue = self.gen_fn_def(free_vars, maybe_return_annotation[0])
            prologue = ''.join(ckpt_func) + prologue
            prologue = prologue

            code = ''.join(body)
            code = '\n'.join('    ' + line for line in code.split('\n'))
            fn_code = f"""
{wrap_stmts}

{prologue}
{code}"""
            return PythonCode(fn_code, globals_)

else:

    def python_code_with_activation_checkpoint(self, root_module: str, namespace: _Namespace) -> PythonCode:
        """
        This method is copied from the _python_code of torch.fx.graph.Graph. Modifications are made so that it can generate
        code for activation checkpoint.
        """
        free_vars: List[str] = []
        body: List[str] = []
        globals_: Dict[str, Any] = {}
        wrapped_fns: Dict[str, None] = {}

        # Wrap string in list to pass by reference
        maybe_return_annotation: List[str] = ['']

        def add_global(name_hint: str, obj: Any):
            """Add an obj to be tracked as a global.

            We call this for names that reference objects external to the
            Graph, like functions or types.

            Returns: the global name that should be used to reference 'obj' in generated source.
            """
            if _is_from_torch(obj) and obj != torch.device:    # to support registering torch.device
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
                return '()'

            typename = _type_repr(o)

            # This is a generic type, e.g. typing.List[torch.Tensor]
            if hasattr(o, '__origin__'):
                origin_type = _origin_type_map.get(o.__origin__, o.__origin__)
                origin_typename = add_global(_type_repr(origin_type), origin_type)

                # Assign global names for each of the inner type variables.
                args = [type_repr(arg) for arg in o.__args__]

                return f'{origin_typename}[{",".join(args)}]'

            # Common case: this is a regular module name like 'foo.bar.baz'
            return add_global(typename, o)

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

        for node in reversed(self.nodes):
            map_arg(node.args, lambda n: register_last_uses(n, node))
            map_arg(node.kwargs, lambda n: register_last_uses(n, node))

        # NOTE: we add a variable to distinguish body and ckpt_func
        def delete_unused_values(user: Node, body):
            """
            Delete values after their last use. This ensures that values that are
            not used in the remainder of the code are freed and the memory usage
            of the code is optimal.
            """
            if user.op == 'placeholder':
                return
            if user.op == 'output':
                body.append('\n')
                return
            nodes_to_delete = user_to_last_uses.get(user, [])
            if len(nodes_to_delete):
                to_delete_str = ' = '.join([repr(n) for n in nodes_to_delete] + ['None'])
                body.append(f';  {to_delete_str}\n')
            else:
                body.append('\n')

        # NOTE: we add a variable to distinguish body and ckpt_func
        def emit_node(node: Node, body):
            maybe_type_annotation = '' if node.type is None else f' : {type_repr(node.type)}'
            if node.op == 'placeholder':
                assert isinstance(node.target, str)
                maybe_default_arg = '' if not node.args else f' = {repr(node.args[0])}'
                free_vars.append(f'{node.target}{maybe_type_annotation}{maybe_default_arg}')
                raw_name = node.target.replace('*', '')
                if raw_name != repr(node):
                    body.append(f'{repr(node)} = {raw_name}\n')
                return
            elif node.op == 'call_method':
                assert isinstance(node.target, str)
                body.append(f'{repr(node)}{maybe_type_annotation} = {_format_target(repr(node.args[0]), node.target)}'
                            f'({_format_args(node.args[1:], node.kwargs)})')
                return
            elif node.op == 'call_function':
                assert callable(node.target)
                # pretty print operators
                if node.target.__module__ == '_operator' and node.target.__name__ in magic_methods:
                    assert isinstance(node.args, tuple)
                    body.append(f'{repr(node)}{maybe_type_annotation} = '
                                f'{magic_methods[node.target.__name__].format(*(repr(a) for a in node.args))}')
                    return
                qualified_name = _get_qualified_name(node.target)
                global_name = add_global(qualified_name, node.target)
                # special case for getattr: node.args could be 2-argument or 3-argument
                # 2-argument: attribute access; 3-argument: fall through to attrib function call with default value
                if global_name == 'getattr' and \
                   isinstance(node.args, tuple) and \
                   isinstance(node.args[1], str) and \
                   node.args[1].isidentifier() and \
                   len(node.args) == 2:
                    body.append(
                        f'{repr(node)}{maybe_type_annotation} = {_format_target(repr(node.args[0]), node.args[1])}')
                    return
                body.append(
                    f'{repr(node)}{maybe_type_annotation} = {global_name}({_format_args(node.args, node.kwargs)})')
                if node.meta.get('is_wrapped', False):
                    wrapped_fns.setdefault(global_name)
                return
            elif node.op == 'call_module':
                assert isinstance(node.target, str)
                body.append(f'{repr(node)}{maybe_type_annotation} = '
                            f'{_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})')
                return
            elif node.op == 'get_attr':
                assert isinstance(node.target, str)
                body.append(f'{repr(node)}{maybe_type_annotation} = {_format_target(root_module, node.target)}')
                return
            elif node.op == 'output':
                if node.type is not None:
                    maybe_return_annotation[0] = f" -> {type_repr(node.type)}"
                if self._pytree_info is None:
                    body.append(f'return {repr(node.args[0])}')
                else:
                    body.append(f'return pytree.tree_unflatten({repr(node.args[0])}, self._out_spec)')
                return
            raise NotImplementedError(f'node: {node.op} {node.target}')

        # Modified for activation checkpointing
        ckpt_func = []
        emit_code_with_activation_checkpoint(body, ckpt_func, self.nodes, emit_node, delete_unused_values)

        if len(body) == 0:
            # If the Graph has no non-placeholder nodes, no lines for the body
            # have been emitted. To continue to have valid Python code, emit a
            # single pass statement
            body.append('pass\n')
        if self._pytree_info is not None:
            orig_args = self._pytree_info.orig_args
            has_orig_self = (orig_args[0] == 'self')
            if has_orig_self:
                free_vars.insert(0, 'self')
            if len(free_vars) > 0:    # pytree has placeholders in it
                body.insert(
                    0,
                    f"{', '.join(free_vars)}, = fx_pytree.tree_flatten_spec([{', '.join(orig_args)}], self._in_spec)\n")
        else:
            orig_args = free_vars

        if len(wrapped_fns) > 0:
            wrap_name = add_global('wrap', torch.fx.wrap)
            wrap_stmts = '\n'.join([f'{wrap_name}("{name}")' for name in wrapped_fns])
        else:
            wrap_stmts = ''

        ckpt_func = ''.join(ckpt_func)

        # If the original function didn't have self as its first argument, we
        # would have added it.
        if len(orig_args) == 0 or orig_args[0] != 'self':
            orig_args.insert(0, 'self')
        code = ''.join(body)
        code = '\n'.join('    ' + line for line in code.split('\n'))

        # as we need colossalai.utils.checkpoint, we need to import colossalai
        # in forward function
        # TODO: Remove inline import
        fn_code = f"""
{wrap_stmts}

{ckpt_func}
def forward({', '.join(orig_args)}){maybe_return_annotation[0]}:
{code}"""
        return PythonCode(fn_code, globals_)
