import torch
from colossalai.tensor.sharding_spec import ShardingSpec, _DimSpec
from colossalai.tensor.utils import all_gather_simulator, all_to_all_simulator, shard_simulator
from enum import Enum
from copy import deepcopy
import math
from functools import reduce
import operator


class CollectiveCommPattern(Enum):
    ALLGATHER = 'all_gather'
    ALLTOALL = 'all_to_all'
    SHARD = 'shard'


class CommSpec:
    '''
    Communication spec is used to record the communication action. It has two main functions:
    1. Compute the communication cost which will be used in auto parallel solver.
    2. Convert the communication spec to real action which will be used in runtime.
    It contains comm_pattern to determine the
    communication method, sharding_spec to determine the communication size, gather_dim and shard_dim 
    to determine the buffer shape, and logical_process_axis

    Argument:
        comm_pattern(CollectiveCommPattern): decribe the communication method used in this spec.
        sharding_spec(ShardingSpec): This is sharding spec of the tensor which will join the communication action.
        gather_dim(int, optional): The gather_dim of the tensor will be gathered.
        shard_dim(int, optional): The shard_dim of the tensor will be sharded.
        logical_process_axis(int, optional): The mesh_dim to implement the communication action.
    '''

    def __init__(self, comm_pattern, sharding_spec, gather_dim=None, shard_dim=None, logical_process_axis=None):
        self.comm_pattern = comm_pattern
        self.sharding_spec = sharding_spec
        self.gather_dim = gather_dim
        self.shard_dim = shard_dim
        self.logical_process_axis = logical_process_axis

    def __repr__(self):
        res_list = ["CommSpec:("]
        if self.comm_pattern == CollectiveCommPattern.ALLGATHER:
            res_list.append(f"comm_pattern:allgather, ")
            res_list.append(f"gather_dim:{self.gather_dim}, ")
            res_list.append(f"logical_process_axis:{self.logical_process_axis})")
        elif self.comm_pattern == CollectiveCommPattern.ALLTOALL:
            res_list.append(f"comm_pattern:all2all, ")
            res_list.append(f"gather_dim:{self.gather_dim}, ")
            res_list.append(f"shard_dim:{self.shard_dim}, ")
            res_list.append(f"logical_process_axis: {self.logical_process_axis})")
        else:
            res_list.append(f"comm_pattern:shard, ")
            res_list.append(f"shard_dim:{self.shard_dim}, ")
            res_list.append(f"logical_process_axis:{self.logical_process_axis})")
        return ''.join(res_list)

    def get_comm_cost(self):
        '''
        For all_gather and all2all operation, the formula provided in DeviceMesh with alpha-beta model is used to
        compute the communication cost.
        For shard operation, it is an on-chip operation, so the communication cost is zero. 
        '''
        comm_size = reduce(operator.mul, self.sharding_spec.get_sharded_shape_per_device(), 1)
        if self.comm_pattern == CollectiveCommPattern.ALLGATHER:
            return self.sharding_spec.device_mesh.all_gather_cost(comm_size, self.logical_process_axis)
        if self.comm_pattern == CollectiveCommPattern.ALLTOALL:
            return self.sharding_spec.device_mesh.all_to_all_cost(comm_size, self.logical_process_axis)
        return 0

    def covert_spec_to_action(self):
        pass


class ShapeConsistencyManager:

    def __init__(self, consistency_option=None):
        self.consistency_option = consistency_option
        self.total_communication_cost = 0
        self.total_transform_steps = 0
        self.cached_spec_pairs_transform_path = {}

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
        comm_pattern = CollectiveCommPattern.ALLGATHER
        for target_pair in source_spec.dim_partition_dict.items():
            shard_list = all_gather_simulator(target_pair)
            index = target_pair[0]
            new_dim_partition_dict = deepcopy(source_spec.dim_partition_dict)

            # We won't add empty list into dim_partition_dict
            # The key will be popped if the related shard_list is empty
            if shard_list:
                new_dim_partition_dict[index] = shard_list
            else:
                new_dim_partition_dict.pop(index)

            # generate the CommSpec to record the action of source_sharding_spec->new_sharding_spec
            gather_dim = index
            logical_process_axis = target_pair[1][-1]
            comm_spec = CommSpec(comm_pattern,
                                 sharding_spec=source_spec,
                                 gather_dim=gather_dim,
                                 logical_process_axis=logical_process_axis)

            # compute the communication cost with CommSpec
            cost = comm_spec.get_comm_cost()

            # generate new sharding spec
            new_sharding_spec = ShardingSpec(source_spec.device_mesh,
                                             source_spec.entire_shape,
                                             dim_partition_dict=new_dim_partition_dict)
            valid_spec_dict[new_sharding_spec] = (comm_spec, orig_cost + cost)
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
        comm_pattern = CollectiveCommPattern.ALLTOALL
        tensor_dims = len(source_spec.entire_shape)
        for f_index in range(tensor_dims - 1):
            for b_index in range(f_index + 1, tensor_dims):
                # skip (R, R) cases
                if f_index not in source_spec.dim_partition_dict and b_index not in source_spec.dim_partition_dict:
                    continue
                else:
                    if f_index in source_spec.dim_partition_dict:
                        # skip (S01, R) -> (R, S01) is NOT allowed
                        if len(source_spec.dim_partition_dict[f_index]) >= 2:
                            continue
                        f_target_pair = (f_index, deepcopy(source_spec.dim_partition_dict[f_index]))
                    else:
                        f_target_pair = (f_index, [])
                    if b_index in source_spec.dim_partition_dict:
                        # skip (R, R) -> (R, S01) is NOT allowed
                        if len(source_spec.dim_partition_dict[b_index]) >= 2:
                            continue
                        b_target_pair = (b_index, deepcopy(source_spec.dim_partition_dict[b_index]))
                    else:
                        b_target_pair = (b_index, [])

                # skip (S1, S0) -> S10
                if f_target_pair[1] and b_target_pair[1] and f_target_pair[1][0] >= b_target_pair[1][0]:
                    continue
                f_shard_list, b_shard_list = all_to_all_simulator(f_target_pair, b_target_pair)
                f_index = f_target_pair[0]
                b_index = b_target_pair[0]

                # generate the CommSpec to record the action of source_sharding_spec->new_sharding_spec
                if len(f_shard_list) < len(f_target_pair[1]):
                    gather_dim = f_index
                    shard_dim = b_index
                    logical_process_axis = f_target_pair[1][-1]
                else:
                    gather_dim = b_index
                    shard_dim = f_index
                    logical_process_axis = b_target_pair[1][-1]
                comm_spec = CommSpec(comm_pattern,
                                     sharding_spec=source_spec,
                                     gather_dim=gather_dim,
                                     shard_dim=shard_dim,
                                     logical_process_axis=logical_process_axis)

                # compute the communication cost with CommSpec
                cost = comm_spec.get_comm_cost()
                new_dim_partition_dict = deepcopy(source_spec.dim_partition_dict)

                # We won't add empty list into dim_partition_dict
                # The key will be popped if the related shard_list is empty
                if f_shard_list:
                    new_dim_partition_dict[f_index] = f_shard_list
                else:
                    new_dim_partition_dict.pop(f_index)
                if b_shard_list:
                    new_dim_partition_dict[b_index] = b_shard_list
                else:
                    new_dim_partition_dict.pop(b_index)

                # generate new sharding spec
                new_sharding_spec = ShardingSpec(source_spec.device_mesh,
                                                 source_spec.entire_shape,
                                                 dim_partition_dict=new_dim_partition_dict)
                valid_spec_dict[new_sharding_spec] = (comm_spec, orig_cost + cost)
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
        comm_pattern = CollectiveCommPattern.SHARD

        # legal sharding dims means the mesh_id is still available to use.
        legal_sharding_dims = [i for i in range(len(source_spec.device_mesh.mesh_shape))]
        for dim, shard_list in source_spec.dim_partition_dict.items():
            for element in shard_list:
                legal_sharding_dims.remove(element)
        if len(legal_sharding_dims) == 0:
            return valid_spec_dict

        tensor_dims = len(source_spec.entire_shape)
        for index in range(tensor_dims):
            if index not in source_spec.dim_partition_dict:
                shard_list_list = shard_simulator((index, []), legal_sharding_dims)
            else:
                shard_list_list = shard_simulator((index, source_spec.dim_partition_dict[index]), legal_sharding_dims)
            if not shard_list_list:
                continue
            for shard_list in shard_list_list:
                new_dim_partition_dict = deepcopy(source_spec.dim_partition_dict)
                new_dim_partition_dict[index] = shard_list

                # generate the CommSpec to record the action of source_sharding_spec->new_sharding_spec
                shard_dim = index
                logical_process_axis = shard_list[-1]
                comm_spec = CommSpec(comm_pattern,
                                     sharding_spec=source_spec,
                                     shard_dim=shard_dim,
                                     logical_process_axis=logical_process_axis)

                # compute the communication cost with CommSpec
                cost = comm_spec.get_comm_cost()

                # generate new sharding spec
                new_sharding_spec = ShardingSpec(source_spec.device_mesh,
                                                 source_spec.entire_shape,
                                                 dim_partition_dict=new_dim_partition_dict)
                valid_spec_dict[new_sharding_spec] = (comm_spec, orig_cost + cost)
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

        During finding the transform path, commucation cost will be accumulated, and it
        will be finally used in auto parallel solver. 

        Additionally, to avoid repeating the path search in runtime, we cached all solved path
        in auto parallel strategy building time, which could handle most of cases in runtime.

        Argument:
            source_spec(ShardingSpec): ShardingSpec of the source activation.
            target_spec(ShardingSpec): ShardingSpec of the target activation.

        Return:
            transform_path(List[ShardingSpec]): The transform path from source_spec to target_spec,
                                                it contains the source_spec and target_spec.
            comm_action_sequence(List[CommSpec]): Keep the communication operations to complete the shape consistency in order.
            total_cost(float): total cost to complete shape consistency transform.

        Example:
            dim_partition_source = {1: [0, 1]}
            dim_partition_target = {0: [0, 1]}
            # DistSpec: 
            #     shard_sequence: R,S01,R 
            #     device_mesh_shape: (4, 4)
            sharding_spec_source = ShardingSpec(device_mesh, entire_shape, dim_partition_source)
            # DistSpec: 
            #     shard_sequence: S01,R,R 
            #     device_mesh_shape: (4, 4)
            sharding_spec_target = ShardingSpec(device_mesh, entire_shape, dim_partition_target)
            transform_path, comm_action_sequence, total_cost = shape_consistency_manager.shape_consistency(sharding_spec_source, sharding_spec_target)
            print(f'transform_path: {transform_path}')
            print(f'comm_action_sequence: {comm_action_sequence}')
            print(f'total_cost: {total_cost}')
    
        output:
            transform_path: [DistSpec: 
                    shard_sequence: R,S01,R 
                    device_mesh_shape: (4, 4), DistSpec: 
                    shard_sequence: R,S0,R 
                    device_mesh_shape: (4, 4), DistSpec: 
                    shard_sequence: S0,R,R 
                    device_mesh_shape: (4, 4), DistSpec: 
                    shard_sequence: S01,R,R 
                    device_mesh_shape: (4, 4)]
            comm_action_sequence: [CommSpec:(comm_pattern:allgather, gather_dim:1, logical_process_axis:1), 
                                   CommSpec:(comm_pattern:all2all, gather_dim:1, shard_dim:0, logical_process_axis: 0),
                                   CommSpec:(comm_pattern:shard, shard_dim:0, logical_process_axis:1)]
            total_cost: 12294.402000000002
        '''
        MAX_TRANSFORM_STEPS = 20
        total_cost = 0
        total_steps = 0
        transform_path = []
        comm_action_sequence = []
        spec_pairs = (str(source_spec.sharding_sequence), str(target_spec.sharding_sequence))
        self.cached_spec_pairs_transform_path[spec_pairs] = (None, None)

        # We do nothing if the sharding spec is all the same.
        if source_spec.sharding_sequence_difference(target_spec) == 0:
            self.cached_spec_pairs_transform_path[spec_pairs] = (transform_path, comm_action_sequence)
            return (transform_path, comm_action_sequence, total_cost)

        temp_sharding_spec = deepcopy(source_spec)
        transform_path.append(temp_sharding_spec)
        # To avoid dead loop, the loop will break after MAX_TRANSFORM_STEPS transforms
        while total_steps <= MAX_TRANSFORM_STEPS:
            valid_transform_spec_dict = self.get_all_one_step_transform_spec(temp_sharding_spec, total_cost)
            best_difference_score = math.inf

            for sharding_spec, info_pairs in valid_transform_spec_dict.items():
                comm_spec, cost = info_pairs
                spec_difference = sharding_spec.sharding_sequence_difference(target_spec)

                if spec_difference == 0:
                    total_cost += cost
                    transform_path.append(sharding_spec)
                    comm_action_sequence.append(comm_spec)
                    self.cached_spec_pairs_transform_path[spec_pairs] = (transform_path, comm_action_sequence)
                    return (transform_path, comm_action_sequence, total_cost)

                if spec_difference < best_difference_score:
                    temp_sharding_spec = deepcopy(sharding_spec)
                    temp_cost = cost
                    temp_comm_spec = deepcopy(comm_spec)
                    best_difference_score = spec_difference

            transform_path.append(temp_sharding_spec)
            comm_action_sequence.append(temp_comm_spec)
            total_cost += temp_cost
            total_steps += 1

        raise RuntimeError(f"Could not find a valid transform path with in {MAX_TRANSFORM_STEPS} steps.")
