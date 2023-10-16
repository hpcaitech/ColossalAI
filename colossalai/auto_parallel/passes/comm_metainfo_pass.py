from typing import Dict

import torch
from torch.fx import GraphModule
from torch.fx.node import Node

from colossalai.auto_parallel.meta_profiler import ShardMetaInfo
from colossalai.auto_parallel.passes.runtime_apply_pass import runtime_apply, runtime_comm_spec_apply
from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, TrainCycleItem
from colossalai.tensor.comm_spec import CommSpec
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec

shape_consistency_manager = ShapeConsistencyManager()


def _construct_shard_meta_info(
    node: Node, origin_sharding_spec: ShardingSpec, target_sharding_spec: ShardingSpec
) -> ShardMetaInfo:
    # get comm_action_sequence and total_cost from shape_consistency_manager
    _, comm_action_sequence, total_cost = shape_consistency_manager.shape_consistency(
        origin_sharding_spec, target_sharding_spec
    )

    meta_info = ShardMetaInfo()
    # NOTE: the cost in shape_consistency_manager.mem_cost is the count in number of numel
    # get mem cost for ShardMetaInfo
    mem_cost = shape_consistency_manager.mem_cost(comm_action_sequence)
    # extract user that has _meta_data and extract element length
    input_node = next(n for n in node._input_nodes if hasattr(n, "_meta_data"))
    element_length = input_node._meta_data.element_size()

    mem_cost.fwd.activation *= element_length
    mem_cost.fwd.temp *= element_length
    mem_cost.bwd.activation *= element_length
    mem_cost.bwd.temp *= element_length
    mem_cost.total.activation *= element_length

    meta_info.memory_cost = mem_cost

    # get computation cost for ShardMetaInfo
    meta_info.compute_cost = TrainCycleItem(
        total_cost["forward"] * element_length,
        total_cost["backward"] * element_length,
        total_cost["total"] * element_length,
    )

    # get tensor shape for ShardMetaInfo
    origin_sharding_spec: ShardingSpec
    target_sharding_spec: ShardingSpec
    input_shape = origin_sharding_spec.get_sharded_shape_per_device()
    output_shape = target_sharding_spec.get_sharded_shape_per_device()

    meta_info.fwd_in = [torch.rand(input_shape, device="meta")]
    meta_info.fwd_buffer = []
    meta_info.fwd_out = [torch.rand(output_shape, device="meta")]

    return meta_info


def _runtime_apply_meta_info(node: Node, origin_spec_dict, sharding_spec_dict) -> ShardMetaInfo:
    """
    This method is used to construct `MetaInto` for shape consistency node
    """

    # extract node index and user node index
    args = node.args
    node_index, user_node_index = args[3], args[4]
    origin_sharding_spec, target_sharding_spec = (
        origin_spec_dict[node_index],
        sharding_spec_dict[node_index][user_node_index],
    )

    return _construct_shard_meta_info(node, origin_sharding_spec, target_sharding_spec)


def _runtime_comm_spec_apply_meta_info(node: Node, comm_actions_dict: Dict) -> ShardMetaInfo:
    # extract node_index and op_data_name
    node_index, op_data_name = node.args[2], node.args[3]

    comm_action = comm_actions_dict[node_index][op_data_name]
    if isinstance(comm_action.comm_spec, CommSpec):
        # this case is for all_reduce, there will be no memory cost
        meta_info = ShardMetaInfo()
        meta_info.memory_cost = TrainCycleItem(MemoryCost(), MemoryCost(), MemoryCost)
        output_node = next(n for n in node.users if hasattr(n, "_meta_data"))
        element_length = output_node._meta_data.element_size()

        total_cost = comm_action.comm_spec.get_comm_cost()
        meta_info.compute_cost = TrainCycleItem(
            total_cost["forward"] * element_length,
            total_cost["backward"] * element_length,
            total_cost["total"] * element_length,
        )

        input_shape = output_shape = comm_action.comm_spec.sharding_spec.get_sharded_shape_per_device()
        meta_info.fwd_in = [torch.rand(input_shape, device="meta")]
        meta_info.fwd_buffer = []
        meta_info.fwd_out = [torch.rand(output_shape, device="meta")]
    else:
        # this case will be handled by shape consistency manager
        origin_sharding_spec, target_sharding_spec = (
            comm_action.comm_spec["src_spec"],
            comm_action.comm_spec["tgt_spec"],
        )
        meta_info = _construct_shard_meta_info(node, origin_sharding_spec, target_sharding_spec)

    return meta_info


def comm_metainfo_pass(
    gm: GraphModule, sharding_spec_dict: Dict, origin_spec_dict: Dict, comm_actions_dict: Dict
) -> GraphModule:
    """
    The method manages all the metainfo of the communication node (run_time_apply, runtime_comm_spec_apply) in the graph.
    """
    for node in gm.graph.nodes:
        if node.target == runtime_apply:
            setattr(node, "best_strategy_info", _runtime_apply_meta_info(node, origin_spec_dict, sharding_spec_dict))
        elif node.target == runtime_comm_spec_apply:
            setattr(node, "best_strategy_info", _runtime_comm_spec_apply_meta_info(node, comm_actions_dict))
        else:
            pass
    return gm
