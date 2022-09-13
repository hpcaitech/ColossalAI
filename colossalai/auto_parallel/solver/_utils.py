import torch
from torch.fx.node import Node
from colossalai.tensor.sharding_spec import ShardingSpec
from colossalai.device.device_mesh import DeviceMesh
from typing import Union, Dict, List


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

    sharding_spec = ShardingSpec(device_mesh=device_mesh, entire_shape=shape, dim_partition_dict=dim_partition_dict)
    return sharding_spec
