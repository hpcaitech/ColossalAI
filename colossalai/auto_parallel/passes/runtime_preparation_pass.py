import operator
from typing import Dict, List, Union

import torch
from torch.fx.node import Node

from colossalai._analyzer.fx.node_util import MetaInfo
from colossalai.auto_parallel.tensor_shard.constants import RESHAPE_FUNC_OP
from colossalai.auto_parallel.tensor_shard.sharding_strategy import CommType, OperationDataType
from colossalai.auto_parallel.tensor_shard.solver.strategies_constructor import StrategiesConstructor
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.comm_spec import _all_reduce
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec

from .constants import SHAPE_ARGUMENT_OPS

shape_consistency_manager = ShapeConsistencyManager()


def size_processing(
    size: Union[int, torch.Size],
    dim_partition_dict: Dict[int, List[int]],
    device_mesh_info: Dict[int, int],
    target_dim: int = None,
    node_name: str = None,
):
    """
    This method will be invoked during runtime to convert size node value depending on distributed information.
    """
    if target_dim is not None:
        assert isinstance(size, int)
        if target_dim in dim_partition_dict:
            total_shard_size = 1
            for shard_dim in dim_partition_dict[target_dim]:
                total_shard_size *= device_mesh_info[shard_dim]
            size = size * total_shard_size

    else:
        size = list(size)
        for dim, dim_size in enumerate(size):
            if dim in dim_partition_dict:
                total_shard_size = 1
                for shard_dim in dim_partition_dict[dim]:
                    total_shard_size *= device_mesh_info[shard_dim]
                size[dim] = dim_size * total_shard_size
        size = torch.Size(size)

    return size


def solution_annotation_pass(
    gm: torch.fx.GraphModule, solution: List[int], strategies_constructor: StrategiesConstructor
):
    """
    This method is used to stick the solution strategy to the nodes and add the information
    required in runtime into graph as placeholder nodes.
    """
    mod_graph = gm.graph

    nodes = [strategies_vector.node for strategies_vector in strategies_constructor.leaf_strategies]
    no_strategy_nodes = strategies_constructor.no_strategy_nodes

    # the dict to get origin sharding spec of node
    origin_node_sharding_spec_dict = {}
    for node_index, (node, strategy_index) in enumerate(zip(nodes, solution)):
        strategies_vector = node.strategies_vector
        # stick the solution strategy to the corresponding node
        setattr(node, "best_strategy", strategies_vector[strategy_index])
        setattr(node, "sharding_spec", strategies_vector[strategy_index].get_sharding_spec_by_name(str(node)))
        origin_node_sharding_spec_dict[node_index] = strategies_vector[strategy_index].get_sharding_spec_by_name(
            str(node)
        )

        # attach the corresponding metainfo if node has the attribute `strategies_info`
        if hasattr(node, "strategies_info"):
            setattr(node, "best_strategy_info", node.strategies_info[strategy_index])

    # the dict to get input sharding specs of user node
    sharding_spec_convert_dict = {}
    # the dict to record comm actions of nodes
    comm_actions_dict = {}
    for index, node in enumerate(nodes):
        target_sharding_specs = []
        for user_node in node.strategies_vector.successor_nodes:
            if user_node in no_strategy_nodes:
                target_sharding_spec = node.best_strategy.get_sharding_spec_by_name(str(node.name))
            else:
                target_sharding_spec = user_node.best_strategy.get_sharding_spec_by_name(str(node.name))
            target_sharding_specs.append(target_sharding_spec)
        sharding_spec_convert_dict[index] = target_sharding_specs
        setattr(node, "target_sharding_specs", target_sharding_specs)

        # the get_attr node strategy is kind of pending strategy, which means we will change it
        # to the same strategy of the user node.
        if node.op == "get_attr":
            assert len(target_sharding_specs) == 1, f"sharing weight is not supported in current version."
            target_node = node.strategies_vector.successor_nodes[0]
            node_name = str(node)
            if target_node.op == "call_function" and target_node.target in RESHAPE_FUNC_OP:
                node_name = str(target_node)
                target_node = target_node.strategies_vector.successor_nodes[0]
            user_strategy = target_node.best_strategy
            op_data_in_user = user_strategy.get_op_data_by_name(node_name)
            origin_pending_strategy = node.best_strategy
            origin_op_data = origin_pending_strategy.get_op_data_by_name(str(node))

            new_communication_actions = {}
            if op_data_in_user in user_strategy.communication_actions:
                new_communication_action = user_strategy.communication_actions.pop(op_data_in_user)
                new_communication_action.arg_index = 0
                new_communication_actions[origin_op_data] = new_communication_action
            node.best_strategy.communication_actions = new_communication_actions

        comm_action_dict = {}
        for op_data, comm_action in node.best_strategy.communication_actions.items():
            comm_action_dict[op_data.name] = comm_action
        comm_actions_dict[index] = comm_action_dict

    # add above dicts into graph
    for node in nodes:
        if node.op != "placeholder":
            with mod_graph.inserting_before(node):
                input_specs_node = mod_graph.create_node("placeholder", target="sharding_spec_convert_dict")
                origin_specs_node = mod_graph.create_node("placeholder", target="origin_node_sharding_spec_dict")
                comm_actions_dict_node = mod_graph.create_node("placeholder", target="comm_actions_dict")
            break
    return gm, sharding_spec_convert_dict, origin_node_sharding_spec_dict, comm_actions_dict


def size_value_converting_pass(gm: torch.fx.GraphModule, device_mesh: DeviceMesh):
    """
    In the auto parallel system, tensors may get shard on different devices, so the size of tensors
    need to be converted to the size of original tensor and managed by the users, such as torch.view,
    torch.reshape, etc. These nodes have enough information like input sharding_spec and
    output sharding_spec to decide how to convert the size value.
    """
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)
    node_pairs = {}

    # DeviceMesh information instructs the scaling of the size value
    device_mesh_info = {}
    for dim, dim_size in enumerate(device_mesh.shape):
        device_mesh_info[dim] = dim_size

    def _extract_target_dim(node):
        """
        A helper function to extract the target dimension from size node.
        There are two usages of torch.Tensor.size:
        1. tensor.size()
        2. tensor.size(dim)

        If a target_dim is assigned, then the output will be in type of int, instead of torch.Size.
        Otherwise, the output will be in type of torch.Size and this function will return None.
        """
        target_dim = None
        if len(node.args) > 1:
            target_dim = node.args[1]
            if target_dim < 0:
                target_dim += node.args[0]._meta_data.dim()
        return target_dim

    def _post_processing(node, size_processing_node):
        """
        This function is used to process the dependency between the size node and its users after
        inserting the size_process_node.
        """
        # store original node and processing node pair in node_pairs dictionary
        # It will be used to replace the original node with processing node in slice object
        node_pairs[node] = size_processing_node
        size_processing_node._meta_data = node._meta_data

        if hasattr(node.meta["info"], "activation_checkpoint"):
            MetaInfo(
                size_processing_node,
                mod_dir=node.meta["info"].mod_dir,
                activation_checkpoint=tuple(node.meta["info"].activation_checkpoint),
            )

        user_list = list(node.users.keys())
        for user in user_list:
            if user == size_processing_node:
                continue
            new_args = list(user.args)
            new_kwargs = dict(user.kwargs)
            # the origin node may be a positional argument or key word argument of user node
            if node in new_args:
                # substitute the origin node with size_processing_node
                new_args[new_args.index(node)] = size_processing_node
                user.args = tuple(new_args)
            elif str(node) in new_kwargs:
                # substitute the origin node with size_processing_node
                new_kwargs[str(node)] = size_processing_node
                user.kwargs = new_kwargs

    def _update_slice_object_args(slice_object):
        """
        This function is used to update the slice object argument list.
        If the slice object contains the Node argument, then the size node will be replaced with
        """
        if isinstance(slice_object, slice):
            start = slice_object.start
            stop = slice_object.stop
            step = slice_object.step
            if start in node_pairs:
                start = node_pairs[start]
            if stop in node_pairs:
                stop = node_pairs[stop]
            if step in node_pairs:
                step = node_pairs[step]
            return slice(start, stop, step)
        elif isinstance(slice_object, int):
            if slice_object in node_pairs:
                return node_pairs[slice_object]
            else:
                return slice_object
        else:
            raise RuntimeError(f"Unsupported slice object type: {type(slice_object)}")

    for node in nodes:
        if node.op == "call_method" and node.target == "size":
            # extract useful information from size node
            # dim_partition_dict will instruct the size value on which
            # dimension should be enlarged.
            sharding_spec = node.args[0].sharding_spec
            dim_partition_dict = sharding_spec.dim_partition_dict

            target_dim = _extract_target_dim(node)

            # insert size_processing node
            with mod_graph.inserting_after(node):
                size_processing_node = mod_graph.create_node(
                    "call_function",
                    size_processing,
                    args=(node, dim_partition_dict, device_mesh_info, target_dim, node.name),
                )
            _post_processing(node, size_processing_node)

        if node.op == "call_function" and node.target == operator.getitem:
            getitem_index = node.args[1]
            # slice object is quite special in torch.fx graph,
            # On one side, we treat slice object same as type of int,
            # so we do not create a node for slice object. On the other side,
            # slice object could take fx.Node as its argument. And the user
            # relationship cannot be tracked in fx graph.
            # Therefore, I record the node_pairs in this pass, and use the it
            # to replace the original node argument inside the slice object if
            # it has been processed in above pass.

            # There are three main usages of operator.getitem:
            #   getitem(input, int)
            #   getitem(input, slice)
            #   getitem(input, Tuple[slice])
            # In this pass, we need process the last two cases because
            # node arguments may potentially appear in these cases.
            if isinstance(getitem_index, slice):
                new_slice_item = _update_slice_object_args(getitem_index)
                new_args = (node.args[0], new_slice_item)
                node.args = new_args

            elif isinstance(getitem_index, (tuple, list)):
                if not isinstance(getitem_index[0], slice):
                    continue
                new_slice_items = []

                for slice_item in getitem_index:
                    if slice_item is None:
                        new_slice_items.append(None)
                        continue
                    new_slice_item = _update_slice_object_args(slice_item)
                    new_slice_items.append(new_slice_item)

                new_args = (node.args[0], tuple(new_slice_items))
                node.args = new_args

    return gm


def node_args_converting_pass(gm: torch.fx.GraphModule, device_mesh: DeviceMesh):
    """
    This pass will process node args to adapt the distributed tensor layout.
    """
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)

    def _extract_info_from_sharding_spec(sharding_spec):
        """
        This function is used to extract the dim_partition_dict and device_mesh from
        sharding spec instance or a list of sharding spec.
        """
        if isinstance(sharding_spec, ShardingSpec):
            dim_partition_dict = sharding_spec.dim_partition_dict
            device_mesh = sharding_spec.device_mesh
            return dim_partition_dict, device_mesh
        if sharding_spec is None:
            return None, None
        assert isinstance(
            sharding_spec, (tuple, list)
        ), "sharding_spec should be type of ShardingSpec, tuple, list or None"

        device_mesh = sharding_spec[0].device_mesh
        dim_partition_dict = []
        for element in sharding_spec:
            dim_partition_dict.append(_extract_info_from_sharding_spec(element))
        return dim_partition_dict, sharding_spec

    def _process_node_arguments(node):
        new_args = []
        for arg in node.args:
            # There are two args style:
            # 1. (input, *shape)
            # 2. (input, shape)
            # We will extract the elements from shape and add them into the new_args
            # Finally, the args style of new_args will be unified to (input, *shape)
            if isinstance(arg, Node):
                if isinstance(arg._meta_data, (tuple, list)):
                    new_args.extend(arg._meta_data)
                elif isinstance(arg._meta_data, int):
                    new_args.append(arg._meta_data)
                else:
                    new_args.append(arg)
            else:
                assert isinstance(
                    arg, (int, tuple, list)
                ), "The argument in view node should be either type of Node or int."
                if isinstance(arg, (tuple, list)):
                    new_args.extend(arg)
                else:
                    new_args.append(arg)
        return new_args

    def _scale_args_adapt_sharding_spec(dim_partition_dict, device_mesh, node):
        new_args = _process_node_arguments(node)
        if node.op == "call_method":
            args_to_process = list(new_args[1:])
        else:
            args_to_process = list(new_args)
        for dim, shard_dims in dim_partition_dict.items():
            total_shard_size = 1
            for shard_dim in shard_dims:
                total_shard_size *= device_mesh.shape[shard_dim]

            # we will skip the dim with -1 value
            if args_to_process[dim] == -1:
                continue
            else:
                # TODO: add assertion here to make sure the dim size is divisible by total_shard_size
                args_to_process[dim] //= total_shard_size

        args_to_process = tuple(args_to_process)

        if node.op == "call_method":
            new_args = (new_args[0],) + args_to_process
        else:
            new_args = args_to_process

        node.args = new_args

    def _filter_node_with_shape_args(node):
        if node.op == "call_method":
            target = getattr(node.args[0]._meta_data.__class__, node.target)
        elif node.op == "call_function":
            target = node.target
        else:
            target = None

        if target in SHAPE_ARGUMENT_OPS:
            return True
        return False

    for node in nodes:
        # skip the placeholder node added in _solution_annotation pass
        if not hasattr(node, "sharding_spec"):
            continue

        output_dim_partition_dict, device_mesh = _extract_info_from_sharding_spec(node.sharding_spec)
        if _filter_node_with_shape_args(node):
            _scale_args_adapt_sharding_spec(output_dim_partition_dict, device_mesh, node)

    return gm


def module_params_sharding_pass(gm: torch.fx.GraphModule, device_mesh: DeviceMesh, overlap=False):
    """
    Apply the sharding action to the module parameters and buffers following the
    instructions of solver solution.
    """
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)
    # This stream is created for overlapping the communication and computation.
    reduction_stream = torch.cuda.Stream()

    def _add_hook_for_grad_communication(node, param, name=None):
        comm_actions = node.best_strategy.communication_actions

        def _filter_param_to_hook(node, op_data, comm_action, name):
            if (
                node.op == "call_module"
                and op_data.type == OperationDataType.PARAM
                and op_data.name == name
                and comm_action.comm_type == CommType.HOOK
            ):
                return True
            if (
                node.op == "get_attr"
                and isinstance(node._meta_data, torch.nn.parameter.Parameter)
                and comm_action.comm_type == CommType.HOOK
            ):
                return True
            return False

        for operation_data, comm_action in comm_actions.items():
            comm_spec_to_use = comm_action.comm_spec
            # register hook to the parameters
            if _filter_param_to_hook(node, operation_data, comm_action, name=name):

                def wrapper(param, comm_spec, stream, overlap):
                    def hook_fn(grad):
                        if overlap:
                            with torch.cuda.stream(stream):
                                _all_reduce(grad, comm_spec, async_op=True)
                        else:
                            _all_reduce(grad, comm_spec, async_op=False)

                    param.register_hook(hook_fn)

                wrapper(param, comm_spec_to_use, reduction_stream, overlap=overlap)

    def _shard_param(param, target_sharding_spec):
        # apply the sharding spec of parameters
        if target_sharding_spec.dim_partition_dict != {}:
            origin_sharding_spec = ShardingSpec(device_mesh, param.shape, {})
            setattr(param, "sharding_spec", origin_sharding_spec)
            # TODO: build a ColoParameter class to manager the distributed parameters
            # we could use .data here, because all the operations just happen before the real training
            # loop, so we don't need to track these operations in the autograd graph.
            param = torch.nn.Parameter(
                shape_consistency_manager.apply_for_autoparallel_runtime(
                    param.data, param.sharding_spec, target_sharding_spec
                )
                .detach()
                .clone()
            )
        return param

    for node in nodes:
        if node.op == "call_module":
            target_module = node.graph.owning_module.get_submodule(node.target)
            # TODO: we need to do more actions to take care of the shared parameters.
            if hasattr(target_module, "processed") and target_module.processed:
                continue
            setattr(target_module, "processed", True)
            for name, param in target_module.named_parameters():
                target_sharding_spec = node.best_strategy.get_sharding_spec_by_name(name)
                param = _shard_param(param, target_sharding_spec)

                setattr(target_module, name, param)
                _add_hook_for_grad_communication(node, param, name)

            sharded_buffer_dict = {}
            # apply the sharding spec of buffers
            for name, buffer in target_module.named_buffers():
                origin_sharding_spec = ShardingSpec(device_mesh, buffer.shape, {})
                setattr(buffer, "sharding_spec", origin_sharding_spec)
                target_sharding_spec = node.best_strategy.get_sharding_spec_by_name(name)
                buffer_sharded = shape_consistency_manager.apply(buffer, target_sharding_spec)
                sharded_buffer_dict[name] = buffer_sharded

            for name, buffer_sharded in sharded_buffer_dict.items():
                setattr(target_module, name, buffer_sharded.detach().clone())

        if node.op == "get_attr":
            root = node.graph.owning_module
            atoms = node.target.split(".")
            attr_len = len(atoms)
            if attr_len == 1:
                target_module = root
                target = getattr(root, atoms[0])
            else:
                target_module = root
                for atom in atoms[:-1]:
                    target_module = getattr(target_module, atom)
                target = getattr(target_module, atoms[-1])

            target_sharding_spec = node.sharding_spec
            target = _shard_param(target, target_sharding_spec)

            assert hasattr(target_module, atoms[-1])
            setattr(target_module, atoms[-1], target)
            _add_hook_for_grad_communication(node, target)

    return gm


def implicit_comm_action_apply(gm: torch.fx.GraphModule):
    """
    replace the origin kernel into kernel with implicit communication inside.
    """


def runtime_preparation_pass(
    gm: torch.fx.GraphModule,
    solution: List[int],
    device_mesh: DeviceMesh,
    strategies_constructor: StrategiesConstructor,
    overlap=False,
):
    gm, sharding_spec_convert_dict, origin_node_sharding_spec_dict, comm_actions_dict = solution_annotation_pass(
        gm, solution, strategies_constructor
    )
    gm = size_value_converting_pass(gm, device_mesh)
    gm = node_args_converting_pass(gm, device_mesh)
    # TODO: the pass below should be uncommented after the implementation of implicit_comm_action_apply_pass completed.
    # gm = implicit_comm_action_apply(gm)
    gm = module_params_sharding_pass(gm, device_mesh, overlap=overlap)

    return gm, sharding_spec_convert_dict, origin_node_sharding_spec_dict, comm_actions_dict
