import torch
from dataclasses import dataclass
from colossalai.tensor.sharding_spec import ShardingSpec, _DimSpec
from colossalai.tensor.utils import all_gather_simulator, all_to_all_simulator, shard_simulator
from enum import Enum
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
from colossalai.context.singleton_meta import SingletonMeta
import torch.distributed as dist
import math
from functools import reduce
import operator
from torch.distributed import ReduceOp

__all__ = [
    'CollectiveCommPattern', 'CommSpec', 'ShapeConsistencyManager', 'ShapeConsistencyOptions',
    'set_shape_consistency_options'
]


def _all_gather(tensor, comm_spec):
    '''
    Implement all gather operation on device mesh based on information provided by comm_spec.
    '''
    process_groups_list = comm_spec.device_mesh.process_groups_dict[comm_spec.logical_process_axis]
    for rank_list, process_group in process_groups_list:
        if dist.get_rank() in rank_list:
            tensor_list = [
                torch.zeros(tensor.shape, dtype=tensor.dtype, device=tensor.device)
                for _ in range(comm_spec.device_mesh.mesh_shape[comm_spec.logical_process_axis])
            ]
            tensor = tensor
            group = process_group
            dist.all_gather(tensor_list, tensor, group=group)
            output = torch.cat(tuple(tensor_list), comm_spec.gather_dim).contiguous()
            return output


def _split(tensor, comm_spec):
    '''
    Implement shard operation on device mesh based on information provided by comm_spec.
    '''
    process_groups_list = comm_spec.device_mesh.process_groups_dict[comm_spec.logical_process_axis]
    for rank_list, _ in process_groups_list:
        if dist.get_rank() in rank_list:
            tensor = tensor
            dim = comm_spec.shard_dim
            length = tensor.shape[comm_spec.shard_dim] // len(rank_list)
            start = length * rank_list.index(dist.get_rank())
            output = torch.narrow(tensor, dim, start, length)
            return output


def _all_to_all(tensor, comm_spec):
    '''
    Implement all to all operation on device mesh based on information provided by comm_spec.
    '''
    process_groups_list = comm_spec.device_mesh.process_groups_dict[comm_spec.logical_process_axis]
    for rank_list, process_group in process_groups_list:
        if dist.get_rank() in rank_list:
            new_shape = list(tensor.shape)
            new_shape[comm_spec.shard_dim] = new_shape[comm_spec.shard_dim] // len(rank_list)
            new_shape = torch.Size(new_shape)
            output_tensor_list = [
                torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device) for _ in range(len(rank_list))
            ]
            dim = comm_spec.shard_dim
            length = tensor.shape[comm_spec.shard_dim] // len(rank_list)
            input_tensor_list = [
                torch.narrow(tensor, dim, length * i, length).contiguous() for i in range(len(rank_list))
            ]
            group = process_group
            dist.all_to_all(output_tensor_list, input_tensor_list, group)
            output = torch.cat(tuple(output_tensor_list), comm_spec.gather_dim).contiguous()
            return output


def _all_reduce(tensor, comm_spec):
    '''
    Implement all reduce operation on device mesh based on information provided by comm_spec.
    '''
    process_groups_list = comm_spec.device_mesh.process_groups_dict[comm_spec.logical_process_axis]
    for rank_list, process_group in process_groups_list:
        if dist.get_rank() in rank_list:
            dist.all_reduce(tensor, op=ReduceOp.SUM, group=process_group)
            return tensor


class _ReduceGrad(torch.autograd.Function):
    """
    A customized communication operation which forward is an identity operation,
    backward is all_reduce operation.

    Args:
        input_: input matrix.
        comm_spec: comm_spec will give information like process group, rank list, etc.
    """

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_, comm_spec):
        ctx.comm_spec = comm_spec
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _all_reduce(grad_output, ctx.comm_spec), None


class _ReduceInput(torch.autograd.Function):
    """
    A customized communication operation which forward is all_reduce operation,
    backward is an identity operation.

    Args:
        input_: input matrix.
        comm_spec: comm_spec will give information like process group, rank list, etc.
    """

    @staticmethod
    def symbolic(graph, input_):
        return _all_reduce(input_)

    @staticmethod
    def forward(ctx, input_, comm_spec):
        return _all_reduce(input_, comm_spec)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    A customized communication operation which forward is split operation,
    backward is an all gather operation.

    Args:
        input_: input matrix.
        comm_spec: comm_spec will give information like process group, rank list, etc.
    """

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_, comm_spec):
        ctx.comm_spec = comm_spec
        return _split(input_, comm_spec)

    @staticmethod
    def backward(ctx, grad_output):
        return _all_gather(grad_output, ctx.comm_spec), None


class _GatherForwardSplitBackward(torch.autograd.Function):
    """
    A customized communication operation which forward is an all gather operation,
    backward is split operation.

    Args:
        input_: input matrix.
        comm_spec: comm_spec will give information like process group, rank list, etc.
    """

    @staticmethod
    def symbolic(graph, input_):
        return _all_gather(input_)

    @staticmethod
    def forward(ctx, input_, comm_spec):
        ctx.comm_spec = comm_spec
        return _all_gather(input_, comm_spec)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.comm_spec), None


class _AllToAll(torch.autograd.Function):
    """
    A customized communication operation which forward is an all to all operation,
    backward is an all to all operation.

    Args:
        input_: input matrix.
        comm_spec: comm_spec will give information like process group, rank list, etc.
    """

    @staticmethod
    def symbolic(graph, input_):
        return _all_to_all(input_)

    @staticmethod
    def forward(ctx, input_, comm_spec):
        output = _all_to_all(input_, comm_spec)
        comm_spec_for_backward = CommSpec(comm_pattern=comm_spec.comm_pattern,
                                          sharding_spec=comm_spec.sharding_spec,
                                          gather_dim=comm_spec.shard_dim,
                                          shard_dim=comm_spec.gather_dim,
                                          logical_process_axis=comm_spec.logical_process_axis)
        ctx.comm_spec = comm_spec_for_backward
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return _all_to_all(grad_outputs, ctx.comm_spec), None


def reduce_grad(input_, comm_spec):
    return _ReduceGrad.apply(input_, comm_spec)


def reduce_input(input_, comm_spec):
    return _ReduceInput.apply(input_, comm_spec)


def split_forward_gather_backward(input_, comm_spec):
    return _SplitForwardGatherBackward.apply(input_, comm_spec)


def gather_forward_split_backward(input_, comm_spec):
    return _GatherForwardSplitBackward.apply(input_, comm_spec)


def all_to_all(input_, comm_spec):
    return _AllToAll.apply(input_, comm_spec)


class CollectiveCommPattern(Enum):
    GATHER_FWD_SPLIT_BWD = 'gather_fwd_split_bwd'
    ALL2ALL_FWD_ALL2ALL_BWD = 'all2all_fwd_all2all_bwd'
    SPLIT_FWD_GATHER_BWD = 'split_fwd_gather_bwd'
    ALLREDUCE_FWD_IDENTITY_BWD = 'all_reduce_fwd_identity_bwd'
    IDENTITY_FWD_ALLREDUCE_BWD = 'identity_fwd_all_reduce_bwd'


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
        gather_dim(int, Optional): The gather_dim of the tensor will be gathered.
        shard_dim(int, Optional): The shard_dim of the tensor will be sharded.
        logical_process_axis(Union(int, List[int]), Optional): The mesh_dim to implement the communication action.
    '''

    def __init__(self,
                 comm_pattern,
                 sharding_spec,
                 gather_dim=None,
                 shard_dim=None,
                 logical_process_axis=None,
                 forward_only=False):
        self.comm_pattern = comm_pattern
        self.sharding_spec = sharding_spec
        self.gather_dim = gather_dim
        self.shard_dim = shard_dim
        self.logical_process_axis = logical_process_axis
        self.forward_only = forward_only
        if isinstance(self.logical_process_axis, list):
            self.device_mesh = self.sharding_spec.device_mesh.flatten_device_mesh
            self.logical_process_axis = 0
        else:
            self.device_mesh = self.sharding_spec.device_mesh

    def __repr__(self):
        res_list = ["CommSpec:("]
        if self.comm_pattern == CollectiveCommPattern.GATHER_FWD_SPLIT_BWD:
            res_list.append(f"comm_pattern:GATHER_FWD_SPLIT_BWD, ")
            res_list.append(f"gather_dim:{self.gather_dim}, ")
            res_list.append(f"logical_process_axis:{self.logical_process_axis})")
        elif self.comm_pattern == CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD:
            res_list.append(f"comm_pattern:ALL2ALL_FWD_ALL2ALL_BWD, ")
            res_list.append(f"gather_dim:{self.gather_dim}, ")
            res_list.append(f"shard_dim:{self.shard_dim}, ")
            res_list.append(f"logical_process_axis: {self.logical_process_axis})")
        elif self.comm_pattern == CollectiveCommPattern.SPLIT_FWD_GATHER_BWD:
            res_list.append(f"comm_pattern:SPLIT_FWD_GATHER_BWD, ")
            res_list.append(f"shard_dim:{self.shard_dim}, ")
            res_list.append(f"logical_process_axis:{self.logical_process_axis})")
        elif self.comm_pattern == CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD:
            res_list.append(f"comm_pattern:ALLREDUCE_FWD_IDENTITY_BWD, ")
            res_list.append(f"logical_process_axis:{self.logical_process_axis})")
        elif self.comm_pattern == CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD:
            res_list.append(f"comm_pattern:IDENTITY_FWD_ALLREDUCE_BWD, ")
            res_list.append(f"logical_process_axis:{self.logical_process_axis})")

        return ''.join(res_list)

    def get_comm_cost(self):
        '''
        For all_gather, all2all, and all_reduce operation, the formula provided in DeviceMesh with alpha-beta model is used to
        compute the communication cost.
        For shard operation, it is an on-chip operation, so the communication cost is zero. 
        '''
        comm_size = reduce(operator.mul, self.sharding_spec.get_sharded_shape_per_device(), 1)
        if self.comm_pattern == CollectiveCommPattern.GATHER_FWD_SPLIT_BWD:
            forward_communication_cost = self.device_mesh.all_gather_cost(comm_size, self.logical_process_axis)
            # give a tiny cost to shard
            backward_communication_cost = 10

        if self.comm_pattern == CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD:
            forward_communication_cost = self.device_mesh.all_to_all_cost(comm_size, self.logical_process_axis)
            # grad should have same shape as input tensor
            # all to all operation has same logical process axis as forward.
            backward_communication_cost = self.device_mesh.all_to_all_cost(comm_size, self.logical_process_axis)

        if self.comm_pattern == CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD:
            forward_communication_cost = self.device_mesh.all_reduce_cost(comm_size, self.logical_process_axis)
            backward_communication_cost = 0

        if self.comm_pattern == CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD:
            forward_communication_cost = 0
            backward_communication_cost = self.device_mesh.all_reduce_cost(comm_size, self.logical_process_axis)

        if self.comm_pattern == CollectiveCommPattern.SPLIT_FWD_GATHER_BWD:
            # give a tiny cost to shard
            forward_communication_cost = 10
            backward_communication_cost = self.device_mesh.all_gather_cost(comm_size, self.logical_process_axis)
        try:
            if self.forward_only:
                total_communication_cost = forward_communication_cost
            else:
                total_communication_cost = forward_communication_cost + backward_communication_cost
        except:
            raise RuntimeError(f"Could not find a matching CollectiveCommPattern for {self.comm_pattern}.")

        return total_communication_cost

    def covert_spec_to_action(self, tensor):
        '''
        Convert CommSpec into runtime action, implement real collection communication to target tensor.
        The collection communication action is directed by the CommSpec.

        Argument:
            tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
        '''
        if self.comm_pattern in pattern_to_func_dict:
            tensor.data = pattern_to_func_dict[self.comm_pattern](tensor, self)
        else:
            tensor.data = tensor


pattern_to_func_dict = {
    CollectiveCommPattern.GATHER_FWD_SPLIT_BWD: gather_forward_split_backward,
    CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD: all_to_all,
    CollectiveCommPattern.SPLIT_FWD_GATHER_BWD: split_forward_gather_backward,
    CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD: reduce_input,
    CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD: reduce_grad,
}


@dataclass
class ShapeConsistencyOptions:
    """
    ShapeConsistencyOptions is a dataclass which specifies the preferences for shape consistency.
    """
    # TODO: shape consistency option is not implemented yet
    pass


def set_shape_consistency_options(options: ShapeConsistencyOptions):
    """
    Configure the shape consistency manager via function call.
    """
    manager = ShapeConsistencyManager()
    manager.options = options


class ShapeConsistencyManager(metaclass=SingletonMeta):

    def __init__(self):
        self._options = None
        self._forward_only = False
        self.total_communication_cost = 0
        self.total_transform_steps = 0
        self.cached_spec_pairs_transform_path = {}

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, options_: ShapeConsistencyOptions):
        assert isinstance(options_, ShapeConsistencyOptions)
        self._options = options_

    @property
    def forward_only(self):
        return self._forward_only

    @forward_only.setter
    def forward_only(self, value):
        assert isinstance(value, bool)
        self._forward_only = value

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
        comm_pattern = CollectiveCommPattern.GATHER_FWD_SPLIT_BWD
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
            comm_spec = CommSpec(
                comm_pattern,
                sharding_spec=source_spec,
                gather_dim=gather_dim,
            # shard_dim will be used during backward
                shard_dim=gather_dim,
                logical_process_axis=logical_process_axis,
                forward_only=self.forward_only)

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
        comm_pattern = CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD
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
                        # skip (R, S01) -> (S01, R) is NOT allowed
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
                                     logical_process_axis=logical_process_axis,
                                     forward_only=self.forward_only)

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
        comm_pattern = CollectiveCommPattern.SPLIT_FWD_GATHER_BWD

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
                                     gather_dim=shard_dim,
                                     shard_dim=shard_dim,
                                     logical_process_axis=logical_process_axis,
                                     forward_only=self.forward_only)

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

        temp_sharding_spec = source_spec
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
                    temp_sharding_spec = sharding_spec
                    temp_cost = cost
                    temp_comm_spec = comm_spec
                    best_difference_score = spec_difference

            transform_path.append(temp_sharding_spec)
            comm_action_sequence.append(temp_comm_spec)
            total_cost += temp_cost
            total_steps += 1

        raise RuntimeError(f"Could not find a valid transform path with in {MAX_TRANSFORM_STEPS} steps.")

    def apply(self, tensor_with_sharding_spec, target_spec):
        '''
        Apply target_spec to tensor with source sharding spec, the transform path is generated by the 
        shape_consistency method.
        
        Argument:
            tensor_with_sharding_spec (torch.Tensor): a tensor with source sharding spec to be transformed to the target spec.
            target_spec (ShardingSpec): The tensor transform processes will be directed by the target_spec.
        
        Example:
            physical_mesh_id = torch.arange(0, 4)
            mesh_shape = (2, 2)
            # [[0, 1,
            #  [2, 3]]
            device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
            entire_shape = torch.Size((4, 2))
            shape_consistency_manager = ShapeConsistencyManager()
            dim_partition_source = {0: [0]}
            dim_partition_target = {1: [0]}

            # DistSpec:
            #     shard_sequence: S0,R
            #     device_mesh_shape: (2, 2)
            sharding_spec_source = ShardingSpec(device_mesh, entire_shape, dim_partition_source)
            
            # DistSpec:
            #     shard_sequence: R,S0
            #     device_mesh_shape: (2, 2)
            sharding_spec_target = ShardingSpec(device_mesh, entire_shape, dim_partition_target)

            if rank in (0, 1):
                sharded_tensor_0 = torch.zeros(2, 1)
                sharded_tensor_1 = torch.ones(2, 1)
                # tensor([[0., 1.],
                #         [0., 1.]])
                tensor_to_comm = torch.cat((sharded_tensor_0, sharded_tensor_1), 1).cuda()
            if rank in (2, 3):
                sharded_tensor_0 = torch.ones(2, 1) * 2
                sharded_tensor_1 = torch.ones(2, 1) * 3
                # tensor([[2., 3.],
                #         [2., 3.]])
                tensor_to_comm = torch.cat((sharded_tensor_0, sharded_tensor_1), 1).cuda()

            tensor_to_comm.sharding_spec = sharding_spec_source
            shape_consistency_manager.apply(tensor_to_comm, sharding_spec_target)
            print(tensor_to_comm)
        
        Output in rank0 and rank2:
            tensor([[0.],
                    [0.],
                    [2.],
                    [2.]])
        
        Output in rank1 and rank3:
            tensor([[1.],
                    [1.],
                    [3.],
                    [3.]])
        '''
        _, comm_action_sequence, _ = self.shape_consistency(tensor_with_sharding_spec.sharding_spec, target_spec)
        for comm_spec in comm_action_sequence:
            comm_spec.covert_spec_to_action(tensor_with_sharding_spec)
        tensor_with_sharding_spec.sharding_spec = target_spec
