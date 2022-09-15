from colossalai.tensor.shape_consistency import ShapeConsistencyManager
import torch
from torch.fx.node import Node
from colossalai.tensor.sharding_spec import ShardingSpec
from colossalai.device.device_mesh import DeviceMesh
from typing import Union, Dict, List, Optional
import warnings
from functools import reduce
import functools
import operator


def generate_sharding_spec(input_: Union[Node, torch.Tensor], device_mesh: DeviceMesh,
                           dim_partition_dict: Dict[int, List[int]]) -> ShardingSpec:
    """
    Generate the sharding spec of the tensor based on the given dim_partition_dict.
    

    Args:
        input_ (Union[Node, torch.Tensor]): the input can be a Node object or a PyTorch tensor. If a node is used, it will look for its meta data associated with this node.
        device_mesh (DeviceMesh): a DeviceMesh object which contains the meta information about the cluster.
        dim_partition_dict (Dict[int, List[int]]): a dictionary to specify the sharding specs, the key is the tensor dimension and the value is the mesh dimension for sharding.
    """

    if isinstance(input_, Node):
        assert hasattr(input_, '_meta_data'), f'The given node has not attribte _meta_data'
        meta_tensor = input_._meta_data
        assert meta_tensor is not None, "The given node's _meta_data attribute is None"
        shape = meta_tensor.shape
    elif isinstance(input_, torch.Tensor):
        shape = input_.shape
    else:
        raise TypeError(
            f'We cannot generate sharding spec for {type(input_)} type, only torch.fx.Node or torch.Tensor is expected.'
        )
    for dim_index, sharding_index_list in dim_partition_dict.items():
        sharding_list = [device_mesh.mesh_shape[sharding_index] for sharding_index in sharding_index_list]
        sharding_size = reduce(operator.mul, sharding_list, 1)
        assert shape[
            dim_index] % sharding_size == 0, f'we cannot shard the {dim_index} dimension of tensor into {sharding_size} partitions.'

    sharding_spec = ShardingSpec(device_mesh=device_mesh, entire_shape=shape, dim_partition_dict=dim_partition_dict)
    return sharding_spec


def generate_resharding_costs(nodes: List[Node],
                              sharding_specs: List[ShardingSpec],
                              count_backward: Optional[bool] = True,
                              dtype: Optional[torch.dtype] = None):
    '''
    Compute the resharding costs with this specific strategy.

    Argument:
        nodes (List[Node]): a list of nodes
        sharding_spec_for_input(ShardingSpec): a list of ShardingSpec for the nodes.
        count_backward (Optional[bool]): whether to include the cost of resharding in the backward pass, default is True. False can be used for inference.
        dtype (Optional[torch.dtype]): the data type for cost calculation, default is None. 
    '''
    # The resharding_cost of weight is counted due to sharing weight cases.
    resharding_costs = {}
    size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()

    # shape consistency manager is a singleton class
    shape_consistency_manager = ShapeConsistencyManager()

    for input_node, input_spec in zip(nodes, sharding_specs):
        resharding_costs[input_node] = []
        for strategy in input_node.strategies_vector:
            input_sharding_spec = strategy.output_sharding_spec
            assert isinstance(input_sharding_spec, ShardingSpec), f'The input node should NOT be a tuple of tensor.'
            # compute the resharding cost during forward phase
            _, _, resharding_cost_forward = shape_consistency_manager.shape_consistency(input_sharding_spec, input_spec)

            if count_backward:
                # In backward phase, we should convert grad with target_spec into input_sharding_spec
                _, _, resharding_cost_backward = shape_consistency_manager.shape_consistency(
                    input_spec, input_sharding_spec)
                total_resharding_cost = resharding_cost_forward + resharding_cost_backward
            else:
                total_resharding_cost = resharding_cost_forward

            # we need multiply the size of elem dtype to get correct communication cost
            resharding_cost = total_resharding_cost * size_per_elem_bytes
            resharding_costs[input_node].append(resharding_cost)
    return resharding_costs


def exception_handler(func):
    """
    A function wrapper which executes the function with a specified seed.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            warnings.warn(f'{e}')

    return wrapper
