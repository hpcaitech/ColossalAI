import torch

from colossalai.tensor.sharding_spec import ShardingSpec


def is_sharding_spec_valid(sharding_spec: ShardingSpec, tensor: torch.Tensor):
    """
    This function checks whether the ShardingSpec is valid for the physical tensor.
    This check includes 2 items:
        1. the sharding spec covers all dimensions of the physical tensor
        2. the sharding spec for each dimension is divisible by the number of devices.
    #
    """
    # make sure all dims are covered in sharding spec
    sharding_len = len(sharding_spec.sharding_sequence)
    tensor_num_dim = tensor.dim()
    num_devices_in_col = sharding_spec.device_mesh.mesh_shape[0]
    num_devices_in_row = sharding_spec.device_mesh.mesh_shape[1]
    assert sharding_len == tensor_num_dim, \
        f'The ShardingSpec ({sharding_spec.sharding_sequence}) is created for {sharding_len}-dimension tensor, but the given tensor is {tensor_num_dim}-dimension ({tensor.shape}).'

    # make sure the sharding is valid for each dim
    for i in range(tensor_num_dim):
        dim_size = tensor.shape[i]
        dim_spec = sharding_spec.sharding_sequence[i]

        if str(dim_spec).startswith('S'):
            devices_str = str(dim_spec).lstrip('S')
            num_devices = 1

            if '0' in devices_str:
                num_devices *= num_devices_in_col
            if '1' in devices_str:
                num_devices *= num_devices_in_row

            assert dim_size >= num_devices and dim_size % num_devices == 0, \
                f'The dimension at index {i} has value {dim_size}, but it is sharded over {num_devices} devices.'
