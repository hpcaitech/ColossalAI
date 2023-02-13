import operator
from copy import deepcopy
from typing import Dict, List, Union

import torch
from torch.fx import symbolic_trace
from torch.fx.node import Node

from colossalai.auto_parallel.tensor_shard.constants import RESHAPE_FUNC_OP
from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    CommAction,
    CommType,
    OperationDataType,
    ShardingStrategy,
)
from colossalai.auto_parallel.tensor_shard.solver.strategies_constructor import StrategiesConstructor
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.comm_spec import _all_reduce
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec

shape_consistency_manager = ShapeConsistencyManager()


def size_processing(size: Union[int, torch.Size],
                    dim_partition_dict: Dict[int, List[int]],
                    device_mesh_info: Dict[int, int],
                    target_dim: int = None,
                    node_name: str = None):
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


def _solution_annotatation(gm: torch.fx.GraphModule,
                           solution: List[int],
                           strategies_constructor: StrategiesConstructor = None):
    """
    This method is used to stick the solution strategy to the nodes and add the information
    required in runtime into graph as placeholder nodes.
    """
    mod_graph = gm.graph
    # TODO: In future PR, strategies_constructor should be a required argument,
    # instead of optional argument. This is because we don't need to consider nodes with
    # no strategy in runtime preparation pass.
    if strategies_constructor is not None:
        nodes = [strategies_vector.node for strategies_vector in strategies_constructor.leaf_strategies]
        no_strategy_nodes = strategies_constructor.no_strategy_nodes
    else:
        nodes = tuple(mod_graph.nodes)
        no_strategy_nodes = []

    # the dict to get origin sharding spec of node
    origin_node_sharding_spec_dict = {}
    for node_index, (node, strategy_index) in enumerate(zip(nodes, solution)):
        strategies_vector = node.strategies_vector
        # stick the solution strategy to the corresponding node
        setattr(node, 'best_strategy', strategies_vector[strategy_index])
        setattr(node, 'sharding_spec', strategies_vector[strategy_index].get_sharding_spec_by_name(str(node)))
        origin_node_sharding_spec_dict[node_index] = strategies_vector[strategy_index].get_sharding_spec_by_name(
            str(node))

        # attach the corresponding metainfo if node has the attribute `metainfo_vector`
        if hasattr(node, 'metainfo_vector'):
            setattr(node, 'best_metainfo', node.metainfo_vector[strategy_index])

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
        setattr(node, 'target_sharding_specs', target_sharding_specs)
        # the get_attr node strategy is kind of pending strategy, which means we will change it
        # to the same strategy of the user node.
        if node.op == 'get_attr':
            assert len(target_sharding_specs) == 1, f'sharing weight is not supported in current version.'
            target_node = node.strategies_vector.successor_nodes[0]
            node_name = str(node)
            if target_node.op == 'call_function' and target_node.target in RESHAPE_FUNC_OP:
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
        if node.op != 'placeholder':
            with mod_graph.inserting_before(node):
                input_specs_node = mod_graph.create_node('placeholder', target='sharding_spec_convert_dict')
                origin_specs_node = mod_graph.create_node('placeholder', target='origin_node_sharding_spec_dict')
                comm_actions_dict_node = mod_graph.create_node('placeholder', target='comm_actions_dict')
            break
    return gm, sharding_spec_convert_dict, origin_node_sharding_spec_dict, comm_actions_dict


def _size_value_converting(gm: torch.fx.GraphModule, device_mesh: DeviceMesh):
    """
    In the auto parallel system, tensors may get shard on different devices, so the size of tensors
    need to be converted to the size of original tensor and managed by the users, such as torch.view,
    torch.reshape, etc. These nodes have enough information like input sharding_spec and
    output sharding_spec to decide how to convert the size value.
    """
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)
    node_pairs = {}

    for node in nodes:

        if node.op == 'call_method' and node.target == 'size':
            # extract useful information from size node
            # dim_partition_dict will instruct the size value on which
            # dimension should be enlarged.
            sharding_spec = node.args[0].sharding_spec
            dim_partition_dict = sharding_spec.dim_partition_dict

            # there are two usages of torch.Tensor.size:
            #   tensor.size()
            #   tensor.size(dim)
            # if a target_dim is assigned, then the output will be
            # in type of int, instead of torch.Size
            target_dim = None
            if len(node.args) > 1:
                target_dim = node.args[1]
                if target_dim < 0:
                    target_dim += node.args[0]._meta_data.dim()

            # DeviceMesh information instructs the scaling of the size value
            device_mesh_info = {}
            for dim, dim_size in enumerate(device_mesh.mesh_shape):
                device_mesh_info[dim] = dim_size

            with mod_graph.inserting_after(node):
                size_processing_node = mod_graph.create_node('call_function',
                                                             size_processing,
                                                             args=(node, dim_partition_dict, device_mesh_info,
                                                                   target_dim, node.name))
                # store original node and processing node pair in node_pairs dictioanry
                # It will be used to replace the original node with processing node in slice object
                node_pairs[node] = size_processing_node
                size_processing_node._meta_data = node._meta_data
                if 'activation_checkpoint' in node.meta:
                    size_processing_node.meta['activation_checkpoint'] = node.meta['activation_checkpoint']

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

        if node.op == 'call_function' and node.target == operator.getitem:

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
                new_start, new_stop, new_step = getitem_index.start, getitem_index.stop, getitem_index.step
                if getitem_index.start in node_pairs:
                    new_start = node_pairs[getitem_index.start]
                elif getitem_index.stop in node_pairs:
                    new_stop = node_pairs[getitem_index.stop]
                elif getitem_index.step in node_pairs:
                    new_step = node_pairs[getitem_index.step]
                new_slice_item = slice(new_start, new_stop, new_step)
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

                    new_start, new_stop, new_step = slice_item.start, slice_item.stop, slice_item.step

                    if slice_item.start in node_pairs:
                        new_start = node_pairs[slice_item.start]
                    elif slice_item.stop in node_pairs:
                        new_stop = node_pairs[slice_item.stop]
                    elif slice_item.step in node_pairs:
                        new_step = node_pairs[slice_item.step]
                    new_slice_item = slice(new_start, new_stop, new_step)
                    new_slice_items.append(new_slice_item)

                new_args = (node.args[0], tuple(new_slice_items))
                node.args = new_args

    return gm


def _node_args_converting(gm: torch.fx.GraphModule, device_mesh: DeviceMesh):
    """
    This pass will process node args to adapt the distributed tensor layout.
    """
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)

    for node in nodes:
        # skip the placeholder node added in _solution_annotation pass
        if not hasattr(node, 'sharding_spec'):
            continue

        def _process_sharding_spec(sharding_spec):
            if isinstance(sharding_spec, ShardingSpec):
                dim_partition_dict = sharding_spec.dim_partition_dict
                device_mesh = sharding_spec.device_mesh
                return dim_partition_dict, device_mesh
            if sharding_spec is None:
                return None, None
            assert isinstance(sharding_spec,
                              (tuple, list)), 'sharding_spec should be type of ShardingSpec, tuple, list or None'

            device_mesh = sharding_spec[0].device_mesh
            dim_partition_dict = []
            for element in sharding_spec:
                dim_partition_dict.append(_process_sharding_spec(element))
            return dim_partition_dict, sharding_spec

        output_dim_partition_dict, device_mesh = _process_sharding_spec(node.sharding_spec)
        new_args = []

        if node.op == 'call_method':
            method = getattr(node.args[0]._meta_data.__class__, node.target)
            # process the node with (input, *shape) style args
            if method in (torch.Tensor.view, torch.Tensor.reshape):

                for arg in node.args:
                    if isinstance(arg, Node):
                        if isinstance(arg._meta_data, (int, tuple, list)):
                            new_args.append(arg._meta_data)
                        else:
                            new_args.append(arg)
                    else:
                        assert isinstance(
                            arg, (int, tuple, list)), 'The argument in view node should be either type of Node or int.'
                        new_args.append(arg)

                for dim, shard_dims in output_dim_partition_dict.items():
                    total_shard_size = 1
                    for shard_dim in shard_dims:
                        total_shard_size *= device_mesh.shape[shard_dim]
                    # There are two ways to use torch.view:
                    # 1. torch.view(input, *shape)
                    # 2. torch.view(input, shape)
                    if isinstance(new_args[1], int):
                        # we will skip the dim with -1 value
                        if new_args[dim + 1] == -1:
                            continue
                        else:
                            new_args[dim + 1] //= total_shard_size
                    else:
                        new_args[1] = list(new_args[1])
                        # we will skip the dim with -1 value
                        if new_args[1][dim] == -1:
                            continue
                        else:
                            new_args[1][dim] //= total_shard_size
                node.args = tuple(new_args)

        elif node.op == 'call_function':
            target = node.target
            # process the node with (input, torch.Size) style args
            if target in (torch.reshape,):
                for arg in node.args:
                    if isinstance(arg, Node):
                        if isinstance(arg._meta_data, (tuple, list)):
                            new_args.append(list(arg._meta_data))
                        else:
                            new_args.append(arg)
                    else:
                        assert isinstance(
                            arg, (tuple, list)), 'The argument in reshape node should be either type of Node or tuple.'
                        new_args.append(list(arg))

                for dim, shard_dims in output_dim_partition_dict.items():
                    # we will skip the dim with -1 value
                    if new_args[1][dim] == -1:
                        continue
                    total_shard_size = 1
                    for shard_dim in shard_dims:
                        total_shard_size *= device_mesh.shape[shard_dim]
                    new_args[1][dim] //= total_shard_size
                node.args = tuple(new_args)

    return gm


def _module_params_sharding(gm: torch.fx.GraphModule, device_mesh: DeviceMesh, overlap=False):
    """
    Apply the sharding action to the module parameters and buffers following the
    instructions of solver solution.
    """
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)
    # This stream is created for overlaping the communication and computation.
    reduction_stream = torch.cuda.Stream()
    for node in nodes:
        if node.op == 'call_module':
            target_module = node.graph.owning_module.get_submodule(node.target)
            # TODO: we need to do more actions to take care of the shared parameters.
            if hasattr(target_module, 'processed') and target_module.processed:
                continue
            setattr(target_module, 'processed', True)
            for name, param in target_module.named_parameters():
                target_sharding_spec = node.best_strategy.get_sharding_spec_by_name(name)
                # apply the sharding spec of parameters
                if target_sharding_spec.dim_partition_dict != {}:
                    origin_sharding_spec = ShardingSpec(device_mesh, param.shape, {})
                    setattr(param, 'sharding_spec', origin_sharding_spec)
                    # TODO: build a ColoParamter class to manager the distributed parameters
                    # we could use .data here, because all the operations just happen before the real training
                    # loop, so we don't need to track these operations in the autograd graph.
                    param.data = shape_consistency_manager.apply_for_autoparallel_runtime(
                        param.data, param.sharding_spec, target_sharding_spec).detach().clone()

                setattr(target_module, name, param)
                comm_actions = node.best_strategy.communication_actions
                for operation_data, comm_action in comm_actions.items():
                    comm_spec_to_use = comm_action.comm_spec
                    # register hook to the parameters
                    if operation_data.type == OperationDataType.PARAM and operation_data.name == name and comm_action.comm_type == CommType.HOOK:

                        def wrapper(param, comm_spec, stream, overlap):

                            def hook_fn(grad):
                                if overlap:
                                    with torch.cuda.stream(stream):
                                        _all_reduce(grad, comm_spec, async_op=True)
                                else:
                                    _all_reduce(grad, comm_spec, async_op=False)

                            param.register_hook(hook_fn)

                        wrapper(param, comm_spec_to_use, reduction_stream, overlap=overlap)

            sharded_buffer_dict = {}
            # apply the sharding spec of buffers
            for name, buffer in target_module.named_buffers():
                origin_sharding_spec = ShardingSpec(device_mesh, buffer.shape, {})
                setattr(buffer, 'sharding_spec', origin_sharding_spec)
                target_sharding_spec = node.best_strategy.get_sharding_spec_by_name(name)
                buffer_sharded = shape_consistency_manager.apply(buffer, target_sharding_spec)
                sharded_buffer_dict[name] = buffer_sharded

            for name, buffer_sharded in sharded_buffer_dict.items():
                setattr(target_module, name, buffer_sharded.detach().clone())

        if node.op == 'get_attr':
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
            if target_sharding_spec.dim_partition_dict != {}:
                origin_sharding_spec = ShardingSpec(device_mesh, target.shape, {})
                setattr(target, 'sharding_spec', origin_sharding_spec)
                # TODO: build a ColoParamter class to manager the distributed parameters
                # we could use .data here, because all the operations just happen before the real training
                # loop, so we don't need to track these operations in the autograd graph.
                target.data = shape_consistency_manager.apply_for_autoparallel_runtime(
                    target.data, target.sharding_spec, target_sharding_spec).detach().clone()

            assert hasattr(target_module, atoms[-1])
            setattr(target_module, atoms[-1], target)

            comm_actions = node.best_strategy.communication_actions
            for operation_data, comm_action in comm_actions.items():
                comm_spec_to_use = comm_action.comm_spec
                # register hook to the parameters
                if isinstance(node._meta_data, torch.nn.parameter.Parameter) and comm_action.comm_type == CommType.HOOK:

                    def wrapper(param, comm_spec, stream, overlap):

                        def hook_fn(grad):
                            if overlap:
                                with torch.cuda.stream(stream):
                                    _all_reduce(grad, comm_spec, async_op=True)
                            else:
                                _all_reduce(grad, comm_spec, async_op=False)

                        param.register_hook(hook_fn)

                    wrapper(target, comm_spec_to_use, reduction_stream, overlap=overlap)
    return gm


def implicit_comm_action_apply(gm: torch.fx.GraphModule):
    """
    replace the origin kernel into kernel with implicit communication inside.
    """
    pass


def runtime_preparation_pass(gm: torch.fx.GraphModule,
                             solution: List[int],
                             device_mesh: DeviceMesh,
                             strategies_constructor: StrategiesConstructor = None,
                             overlap=False):
    gm, sharding_spec_convert_dict, origin_node_sharding_spec_dict, comm_actions_dict = _solution_annotatation(
        gm, solution, strategies_constructor)
    gm = _size_value_converting(gm, device_mesh)
    gm = _node_args_converting(gm, device_mesh)
    # TODO: the pass below should be uncommented after the implementation of implicit_comm_action_apply_pass completed.
    # gm = implicit_comm_action_apply(gm)
    gm = _module_params_sharding(gm, device_mesh, overlap=overlap)

    return gm, sharding_spec_convert_dict, origin_node_sharding_spec_dict, comm_actions_dict
