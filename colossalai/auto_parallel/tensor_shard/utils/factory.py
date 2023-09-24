import copy
import operator
import warnings
from functools import reduce
from typing import Dict, List, Optional, Union

import torch
from torch.fx.node import Node
from torch.utils._pytree import tree_map

from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec

from ..constants import INFINITY_COST

__all__ = ["generate_sharding_spec", "generate_resharding_costs"]


def generate_sharding_spec(
    input_: Union[Node, torch.Tensor], device_mesh: DeviceMesh, dim_partition_dict: Dict[int, List[int]]
) -> ShardingSpec:
    """
    Generate the sharding spec of the tensor based on the given dim_partition_dict.


    Args:
        input_ (Union[Node, torch.Tensor]): the input can be a Node object or a PyTorch tensor. If a node is used, it will look for its meta data associated with this node.
        device_mesh (DeviceMesh): a DeviceMesh object which contains the meta information about the cluster.
        dim_partition_dict (Dict[int, List[int]]): a dictionary to specify the sharding specs, the key is the tensor dimension and the value is the mesh dimension for sharding.
    """

    if isinstance(input_, Node):
        assert hasattr(input_, "_meta_data"), f"The given node has no attribute _meta_data"
        meta_tensor = input_._meta_data
        assert meta_tensor is not None, "The given node's _meta_data attribute is None"
        shape = meta_tensor.shape
    elif isinstance(input_, torch.Tensor):
        shape = input_.shape
    else:
        raise TypeError(
            f"We cannot generate sharding spec for {type(input_)} type, only torch.fx.Node or torch.Tensor is expected."
        )
    for dim_index, sharding_index_list in dim_partition_dict.items():
        sharding_list = [device_mesh.mesh_shape[sharding_index] for sharding_index in sharding_index_list]
        sharding_size = reduce(operator.mul, sharding_list, 1)
        assert (
            shape[dim_index] % sharding_size == 0
        ), f"we cannot shard the {dim_index} dimension of tensor into {sharding_size} partitions."

    sharding_spec = ShardingSpec(device_mesh=device_mesh, entire_shape=shape, dim_partition_dict=dim_partition_dict)
    return sharding_spec


def generate_resharding_costs(
    nodes: List[Node],
    sharding_specs: List[ShardingSpec],
    count_backward: Optional[bool] = True,
    dtype: Optional[torch.dtype] = None,
    index=None,
):
    """
    Compute the resharding costs with this specific strategy.

    Argument:
        nodes (List[Node]): a list of nodes
        sharding_spec_for_input(ShardingSpec): a list of ShardingSpec for the nodes.
        count_backward (Optional[bool]): whether to include the cost of resharding in the backward pass, default is True. False can be used for inference.
        dtype (Optional[torch.dtype]): the data type for cost calculation, default is None.
    """
    # The resharding_cost of weight is counted due to sharing weight cases.
    resharding_costs = {}
    size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()

    # shape consistency manager is a singleton class
    shape_consistency_manager = ShapeConsistencyManager()

    for input_node, input_spec in zip(nodes, sharding_specs):
        resharding_costs[input_node] = []
        for strategy in input_node.strategies_vector:
            input_sharding_spec = strategy.output_sharding_spec
            if not isinstance(input_sharding_spec, ShardingSpec):
                assert isinstance(input_sharding_spec, list), "only ShardingSpec or List[ShardingSpec] is expected."
                input_sharding_spec = input_sharding_spec[index]
            assert isinstance(input_sharding_spec, ShardingSpec), f"The input node should NOT be a tuple of tensor."
            try:
                # compute the resharding cost
                _, _, total_resharding_cost = shape_consistency_manager.shape_consistency(
                    input_sharding_spec, input_spec
                )

                # we need multiply the size of elem dtype to get correct communication cost
                resharding_cost = total_resharding_cost["total"] * size_per_elem_bytes
            except AssertionError as e:
                warnings.warn(f"{e}")
                resharding_cost = INFINITY_COST
            resharding_costs[input_node].append(resharding_cost)
    return resharding_costs


def find_repeat_blocks(node_list: List[torch.fx.Node], root_module, common_length_threshold: int = 20):
    """
    Find the largest repeat blocks in the graph, whose length is larger than the threshold.

    Args:
        gm (GraphModule): the graph module to be analyzed.
        common_length_threshold (int): the threshold of the repeat block length.
    """

    # graph = gm.graph

    def _process_args(args):
        new_args = []
        for arg in args:
            if hasattr(arg, "_meta_data"):
                meta_data = arg._meta_data
            else:
                meta_data = arg

            def _process_arg(data):
                if isinstance(data, torch.Tensor):
                    data = data.size()
                elif isinstance(data, slice):
                    data = (data.start, data.step, data.stop)
                return data

            new_meta_data = tree_map(_process_arg, meta_data)
            new_args.append(new_meta_data)

        return new_args

    def _all_equal(check_list, check_fn):
        base_value = check_list[-1]
        for e in check_list:
            if not check_fn(e, base_value):
                return False
        return True

    def _check_node_list_equal(l1, l2):
        if len(l1) != len(l2):
            return False
        for node1, node2 in zip(l1, l2):
            if hash(node1.hash_key) != hash(node2.hash_key):
                return False
        return True

    def _check_node_equal(node1, node2):
        if hash(node1.hash_key) == hash(node2.hash_key):
            return True
        return False

    for index, node in enumerate(node_list):
        if node.op == "call_module":
            target = node.target
            submod = root_module.get_submodule(target)
            submod_type = type(submod)
            target = submod_type
        else:
            target = node.target

        new_args = _process_args(node.args)

        if node.op != "get_attr":
            hash_key = (node.op, target, *new_args)
        else:
            hash_key = (node.op,)

        setattr(node, "hash_key", hash_key)

    hash_value_to_node_dict = {}

    for index, node in enumerate(node_list):
        hash_value = hash(node.hash_key)
        if hash_value not in hash_value_to_node_dict:
            hash_value_to_node_dict[hash_value] = []
        hash_value_to_node_dict[hash_value].append(index)

    # node_list = list(graph.nodes)

    node_list_start = 0
    max_common_length = common_length_threshold
    common_blocks_index = []
    for index, node in enumerate(node_list):
        # the comparison will be triggered if a common node appears
        if len(hash_value_to_node_dict[hash(node.hash_key)]) >= 2:
            start_index_list = hash_value_to_node_dict[hash(node.hash_key)]
            check_block_list = [node_list[start : start + max_common_length] for start in start_index_list]

            common_label = True
            if not _all_equal(check_block_list, _check_node_list_equal):
                common_label = False

            if common_label:
                common_blocks_index = copy.deepcopy(start_index_list)
                max_step = len(node_list) - common_blocks_index[-1] - max_common_length - 1

                for i in range(max_step):
                    # add assertion to avoid out of index
                    next_node_list = [node_list[index + max_common_length + i] for index in start_index_list]
                    if not _all_equal(next_node_list, _check_node_equal):
                        max_step = i
                        break
                max_common_length += max_step
                node_list_start += max_common_length

    # recover common subgraph from the index
    common_blocks = []
    for start in common_blocks_index:
        common_blocks.append(node_list[start : start + max_common_length])

    return common_blocks
