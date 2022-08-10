import torch
from colossalai.tensor.sharding_spec import ShardingSpec, _DimSpec
from enum import Enum
from copy import deepcopy


class CollectiveCommPattern(Enum):
    ALLGATHER = 'all_gather'
    ALLTOALL = 'all_to_all'
    SHARD = 'shard'


class ShapeConsistencyManager:

    def __init__(self, consistency_option=None):
        self.consistency_option = consistency_option
        self.total_communication_cost = 0
        self.total_transform_steps = 0
        self.cached_spec_pairs = {}

    def _all_gather_simulator(self, target_pair):
        '''
        Simulating all-gather operation, analyze the communication cost
        and simulate the influence of the DimSpec.

        We don't allow uncontiguous layout, such as all-gather(S012)->S02 is NOT allowed.
        Therefore, all gather operation just remove the last element in shard list,
        e.g.: 
            all-gather(S01) -> S0

        Argument:
            target_pair(Tuple[int, List[int]]): The first element is the dimension of tensor to be sharded,
            and the second element decribes which logical axis will be sharded in that dimension.
        '''
        _, shard_list = target_pair
        new_shard_list = shard_list[:-1]
        # TODO: compute comm cost
        comm_cost = 0
        return new_shard_list, comm_cost

    def _all_to_all_simulator(self, f_target_pair, b_target_pair):
        '''
        Simulating all-to-all operation, analyze the communication cost
        and simulate the influence of the DimSpec.

        We BANNED all representations which shard_list in decreasing order,
        such as S10, so all-to-all(S0, S1) -> RS01 is NOT allowed. 
        Therefore, if the behind shard_list is not None, we just extend it to the front shard_list.
        Argument:
            target_pair(Tuple[int, List[int]]): The first element is the dimension of tensor to be sharded,
            and the second element decribes which logical axis will be sharded in that dimension.
        e.g.: 
            all-to-all(S0, S1) -> [S01, R]
            all-to-all(S0, R) -> [R, S0]
        Otherwise, we extend the front shard_list to behind.
        e.g.: 
            all-to-all(R, S1) -> [S1, R]
        
        Argument:
            target_pair(Tuple[int, List[int]]): The first element is the dimension of tensor to be sharded,
            and the second element decribes which logical axis will be sharded in that dimension.
        '''
        _, f_shard_list = f_target_pair
        _, b_shard_list = b_target_pair
        if not len(b_shard_list):
            b_shard_list.extend(f_shard_list)
            f_shard_list = []
        else:
            f_shard_list.extend(b_shard_list)
            b_shard_list = []
        # TODO: compute comm cost
        comm_cost = 0
        return f_shard_list, b_shard_list, comm_cost

    def _shard_simulator(self, target_pair, legal_sharding_dims):
        '''
        Simulating shard operation, analyze the communication cost(always ZERO)
        and simulate the influence of the DimSpec.

        We don't allow uncontiguous layout, such as shard(S0)->S02 is NOT allowed.
        In addition, We BANNED all representations which shard_list in decreasing order, 
        such as S10, so shard(S0) -> S10 is NOT allowed.
        Therefore, for the R dimension, we could just append any legal sharding dim on it.
        e.g.:
            shard(R) -> S0
        For the S dimension, we need to make sure the shard_list after sharding still keep rising order.
        e.g:
            shard(S0) -> S01

        Argument:
            target_pair(Tuple[int, List[int]]): The first element is the dimension of tensor to be sharded,
            and the second element decribes which logical axis will be sharded in that dimension.
        '''
        _, shard_list = target_pair
        shard_list_list = []
        for dim in legal_sharding_dims:
            if len(shard_list) != 0 and dim <= shard_list[-1]:
                continue
            new_shard_list = shard_list + [dim]
            shard_list_list.append(new_shard_list)
        comm_cost = 0
        return shard_list_list, comm_cost

    def get_all_all_gather_spec(self, source_spec, orig_cost):
        '''
        Get all valid sharding specs from source_spec with single all-gather operation, and 
        accumulate commucation cost on origin cost which will finally be used in auto sharding solver.
        For the all-gather operation, we just care about the S dimension.
        
        Argument:
            source_spec(ShardingSpec): the ShardingSpec of the source_spec.
            orig_cost(float): the original communication cost before this operation.

        Return:
            valid_spec_dict(Dict[ShardingSpec, float]): all valid sharding specs from source_spec with single all-gather operation.

        Example:
            dim_partition_dict = {0: [0], 1: [1]}
            # DistSpec:
            #     shard_sequence: S0,S1,R
            #     device_mesh_shape: (4, 4)
            sharding_spec = ShardingSpec(device_mesh, entire_shape, dim_partition_dict)
            shape_consistency_manager = ShapeConsistencyManager()
            rst_dict = shape_consistency_manager.get_all_all_gather_spec(sharding_spec, 0)
            print(rst_dict)
        
        Output:
            {DistSpec: 
            shard_sequence: R,S1,R 
            device_mesh_shape: (4, 4): 0, DistSpec: 
            shard_sequence: S0,R,R 
            device_mesh_shape: (4, 4): 0}
        '''
        valid_spec_dict = {}
        for target_pair in source_spec.dim_partition_dict.items():
            shard_list, cost = self._all_gather_simulator(target_pair)
            index = target_pair[0]
            new_dim_partition_dict = deepcopy(source_spec.dim_partition_dict)
            new_dim_partition_dict[index] = shard_list
            new_sharding_spec = ShardingSpec(source_spec.device_mesh,
                                             source_spec.entire_shape,
                                             dim_partition_dict=new_dim_partition_dict)
            valid_spec_dict[new_sharding_spec] = orig_cost + cost
        return valid_spec_dict

    def get_all_all_to_all_spec(self, source_spec, orig_cost):
        '''
        Get all valid sharding specs from source_spec with single all-to-all operation, and 
        accumulate commucation cost on origin cost which will finally be used in auto sharding solver.
        For the all-to-all operation, we just care about the pairs containing S dimension.
        
        Argument:
            source_spec(ShardingSpec): the ShardingSpec of the source_spec.
            orig_cost(float): the original communication cost before this operation.

        Return:
            valid_spec_dict(Dict[ShardingSpec, float]): all valid sharding specs from source_spec with single all-to-all operation.

        Example:
            dim_partition_dict = {0: [0], 1: [1]}
            # DistSpec:
            #     shard_sequence: S0,S1,R
            #     device_mesh_shape: (4, 4)
            sharding_spec = ShardingSpec(device_mesh, entire_shape, dim_partition_dict)
            shape_consistency_manager = ShapeConsistencyManager()
            rst_dict = shape_consistency_manager.get_all_all_to_all_spec(sharding_spec, 0)
            print(rst_dict)
        
        Output:
            {DistSpec: 
            shard_sequence: S01,R,R 
            device_mesh_shape: (4, 4): 0, DistSpec: 
            shard_sequence: R,S1,S0 
            device_mesh_shape: (4, 4): 0, DistSpec: 
            shard_sequence: S0,R,S1 
            device_mesh_shape: (4, 4): 0}
        '''
        valid_spec_dict = {}
        tensor_dims = len(source_spec.entire_shape)
        for f_index in range(tensor_dims - 1):
            for b_index in range(f_index + 1, tensor_dims):
                # skip (R, R) cases
                if f_index not in source_spec.dim_partition_dict and b_index not in source_spec.dim_partition_dict:
                    continue
                else:
                    if f_index in source_spec.dim_partition_dict:
                        f_target_pair = (f_index, deepcopy(source_spec.dim_partition_dict[f_index]))
                    else:
                        f_target_pair = (f_index, [])
                    if b_index in source_spec.dim_partition_dict:
                        b_target_pair = (b_index, deepcopy(source_spec.dim_partition_dict[b_index]))
                    else:
                        b_target_pair = (b_index, [])

                f_shard_list, b_shard_list, cost = self._all_to_all_simulator(f_target_pair, b_target_pair)
                f_index = f_target_pair[0]
                b_index = b_target_pair[0]
                new_dim_partition_dict = deepcopy(source_spec.dim_partition_dict)
                new_dim_partition_dict[f_index] = f_shard_list
                new_dim_partition_dict[b_index] = b_shard_list
                new_sharding_spec = ShardingSpec(source_spec.device_mesh,
                                                 source_spec.entire_shape,
                                                 dim_partition_dict=new_dim_partition_dict)
                valid_spec_dict[new_sharding_spec] = orig_cost + cost
        return valid_spec_dict

    def get_all_shard_spec(self, source_spec, orig_cost):
        '''
        Get all valid sharding specs from source_spec with single shard operation, and 
        accumulate commucation cost on origin cost which will finally be used in auto sharding solver.
        For the sharding operation, we just care about legal sharding dimensions.
        
        Argument:
            source_spec(ShardingSpec): the ShardingSpec of the source_spec.
            orig_cost(float): the original communication cost before this operation.

        Return:
            valid_spec_dict(Dict[ShardingSpec, float]): all valid sharding specs from source_spec with single all-to-all operation.

        Example:
            dim_partition_dict = {0: [0]}
            # DistSpec:
            #     shard_sequence: S0,R,R
            #     device_mesh_shape: (4, 4)
            sharding_spec = ShardingSpec(device_mesh, entire_shape, dim_partition_dict)
            shape_consistency_manager = ShapeConsistencyManager()
            rst_dict = shape_consistency_manager.get_all_shard_spec(sharding_spec, 0)
            print(rst_dict)
        
        Output:
            {DistSpec: 
            shard_sequence: S01,R,R 
            device_mesh_shape: (4, 4): 0, DistSpec: 
            shard_sequence: S0,S1,R 
            device_mesh_shape: (4, 4): 0, DistSpec: 
            shard_sequence: S0,R,S1 
            device_mesh_shape: (4, 4): 0}
        '''
        valid_spec_dict = {}
        legal_sharding_dims = [i for i in range(len(source_spec.device_mesh.mesh_shape))]
        for dim, shard_list in source_spec.dim_partition_dict.items():
            for element in shard_list:
                legal_sharding_dims.remove(element)
        if len(legal_sharding_dims) == 0:
            return valid_spec_dict

        tensor_dims = len(source_spec.entire_shape)
        for index in range(tensor_dims):
            if index not in source_spec.dim_partition_dict:
                shard_list_list, cost = self._shard_simulator((index, []), legal_sharding_dims)
            else:
                shard_list_list, cost = self._shard_simulator((index, source_spec.dim_partition_dict[index]),
                                                              legal_sharding_dims)
            if not shard_list_list:
                continue
            for shard_list in shard_list_list:
                new_dim_partition_dict = deepcopy(source_spec.dim_partition_dict)
                new_dim_partition_dict[index] = shard_list
                new_sharding_spec = ShardingSpec(source_spec.device_mesh,
                                                 source_spec.entire_shape,
                                                 dim_partition_dict=new_dim_partition_dict)
                valid_spec_dict[new_sharding_spec] = orig_cost + cost
        return valid_spec_dict

    def get_all_one_step_transform_spec(self, source_spec, orig_cost):
        '''
        Get all valid sharding specs from source_spec with one step transform, and 
        accumulate commucation cost on origin cost which will finally be used in auto sharding solver.
        Note:
            all-gather will eliminate a sharding dimension, all-to-all will keep sharding dimension same as before,
            and shard will add a sharding dimension. Therefore, the result of above operations are mutual exclusive,
            we could safely put them together.
        
        Argument:
            source_spec(ShardingSpec): the ShardingSpec of the source_spec.
            orig_cost(float): the original communication cost before this operation.

        Return:
            valid_spec_dict(Dict[ShardingSpec, float]): all valid sharding specs from source_spec with single all-to-all operation.
        '''
        valid_spec_dict = {}
        valid_spec_dict.update(self.get_all_all_gather_spec(source_spec, orig_cost))
        valid_spec_dict.update(self.get_all_all_to_all_spec(source_spec, orig_cost))
        valid_spec_dict.update(self.get_all_shard_spec(source_spec, orig_cost))
        return valid_spec_dict

    def shape_consistency(self, source_spec, target_spec):
        '''
        This method will find a path to transform source_spec to target_spec with
        a greedy algorithm.
        The basic idea is:
        Step1:
            Generate all one-step transform sequences from source_spec.
        Step2:
            Pick the 'best' sharding spec following the heuristic function.
        Step3:
            Repeat above steps until the source spec transform to target spec.

        This function is NOT completed, due to absense of difference function.
        '''
        MAX_TRANSFORM_STEPS = 10
        total_cost = 0
        total_steps = 0
        transform_path = []
        temp_sharding_spec = deepcopy(source_spec)
        transform_path.append(temp_sharding_spec)
        while total_steps <= MAX_TRANSFORM_STEPS:
            valid_transform_spec_dict = get_all_one_step_transform_spec(temp_sharding_spec)
            best_difference_score = 0
            for sharding_spec, cost in valid_transform_spec_dict.items():
                if no_difference(sharding_spec, target_spec):
                    total_cost += cost
                    transform_path.append(sharding_spec)
                    return (transform_path, total_cost)
                if difference(sharding_spec, target_spec) > best_difference_score:
                    temp_sharding_spec = deepcopy(sharding_spec)
                    temp_cost = cost
            transform_path.append(temp_sharding_spec)
            total_cost += temp_cost
        return (transform_path, total_cost)
