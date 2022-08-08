from colossalai.device.device_mesh import DeviceMesh


class _DimSpec:
    '''
    Sharding spec for single dimension of the sharded tensor decribe the sharding dimension of
    logical device mesh and give a method to compute the difference between them.
    This class is used internally in ShardingSpec.

    Argument:
        shard_list(List[int]): if shard_list is None, the dim spec will be 'R' type. 
            Otherwise, the element in shard_list means the data will be sharded in that dimension.
    '''

    def __init__(self, shard_list):
        self.is_replica = shard_list is None
        self.shard_list = shard_list

    def __eq__(self, other):
        if dir(self) != dir(other):
            return False
        for attr in dir(self):
            if not attr.startswith('__') and getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def __repr__(self):
        if self.is_replica:
            return 'R'
        target = 'S'
        for dim in self.shard_list:
            target += str(dim)
        return target

    def difference(self, other):
        '''
        This function is temporarily NOT implemented, it will be codesigned with ShapeConsistency feature.
        '''
        pass


class ShardingSpec:
    '''
    Sharding spec for a tensor, it contains info of the logical device mesh this tensor belong
    to, the entire shape of the tensor before sharded, and the sharding sequence looks like 
    [R, R, S0, S1].
    
    Argument:
        device_mesh(DeviceMesh): A logical view of a physical mesh.
        entire_shape(torch.Size): The entire shape of tensor before sharded.
        dim_partition_dict(Dict[int, List[int]]): The key is the dimension of tensor to be sharded,
            and the value of the key decribe which logical axis will be sharded in that dimension.
    '''

    def __init__(self, device_mesh, entire_shape, dim_partition_dict):
        self.device_mesh = device_mesh
        self.entire_shape = entire_shape
        self.dim_partition_dict = dim_partition_dict
        self._sanity_check()
        self.sharding_sequence = self.convert_dict_to_shard_sequence()

    def __repr__(self):
        res_list = ["DistSpec:"]
        res_list.append(f"\n\tshard_sequence: " + ",".join(str(dimspec) for dimspec in self.sharding_sequence))
        res_list.append(f"\n\tdevice_mesh_shape: {self.device_mesh.mesh_shape}")
        return ' '.join(res_list)

    def _sanity_check(self):
        '''
        In sanity check, we need make sure all axes in logical device mesh only be used
        once.
        '''
        dim_check_list = [i for i in range(self.device_mesh.logical_mesh_id.dim())]
        for dim, shard_list in self.dim_partition_dict.items():
            for element in shard_list:
                if element in dim_check_list:
                    dim_check_list.remove(element)
                else:
                    raise ValueError(
                        f"find an invalid sharding axis {element} in dim_partition_dict in tensor dimension {dim}.")

    def convert_dict_to_shard_sequence(self):
        sharding_sequence = [_DimSpec(None)] * len(self.entire_shape)
        for dim, shard_list in self.dim_partition_dict.items():
            sharding_sequence[dim] = _DimSpec(shard_list)
        return sharding_sequence

    def sharding_sequence_difference(self, other):
        '''
        This function is temporarily NOT implemented, it will be codesigned with ShapeConsistency feature.
        '''
        pass
