import operator
from enum import Enum
from functools import reduce
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from colossalai.auto_parallel.tensor_shard.deprecated._utils import ignore_sharding_exception
from colossalai.auto_parallel.tensor_shard.deprecated.sharding_strategy import ShardingStrategy, StrategiesVector

from ..constants import LINEAR_FUNC_OP, LINEAR_MODULE_OP
from .operator_handler import OperatorHandler
from .strategy_generator import IntermediateStrategy, StrategyGenerator

__all__ = ['DotHandler']


class DotProductStrategyGenerator(StrategyGenerator):
    """
    DotProductStrategyGenerator is used to generate the sharding strategies for two 1D tensors in dot product computation.
    This is created for torch.matmul where two tensors are 1D tensors. As torch.matmul does not include a bias argument, so we
    do not consider bias here.
    """

    def validate(self, input, other):
        assert input.dim() == 1 and other.dim() == 1

    def no_split(self):
        name = f'R = R dot R'
        dim_partition_dict = {"input": {}, "other": {}, "output": {}}
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict)

    def split_one_dim(self, mesh_dim):
        name = f'S{mesh_dim} = S{mesh_dim} dot S{mesh_dim}'
        dim_partition_dict = {"input": {0: [mesh_dim]}, "other": {0: [mesh_dim]}, "output": {}}
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict, all_reduce_axis=[mesh_dim])

    def generate(self) -> List[IntermediateStrategy]:
        strategy_list = []

        # do not split dimensions for dot product
        # R = R dot R
        strategy_list.append(self.no_split())

        # split two tensors in the same dimensions
        # S = S dot S
        strategy_list.append(self.split_one_dim(0))
        strategy_list.append(self.split_one_dim(1))

        return strategy_list


class MatVecStrategyGenerator(StrategyGenerator):

    def validate(self, input, other) -> bool:
        assert input.dim() > 1 and other.dim() == 1

    def no_split(self):
        name = "R = R x R"
        dim_partition_dict = {"input": {}, "other": {}, "output": {}}
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict)

    def split_input_batch(self, mesh_dim):
        name = f'S{mesh_dim}R = S{mesh_dim}R x R'
        dim_partition_dict = {"input": {0: [mesh_dim]}, "other": {}, "output": {0: [mesh_dim]}}
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict)

    def generate(self) -> List[IntermediateStrategy]:
        strategy_list = []

        # no split
        strategy_list.append(self.no_split())

        # split the batch dim for the first tensor only
        strategy_list.append(self.split_input_batch(0))
        strategy_list.append(self.split_input_batch(1))

        return strategy_list


class MatMulStrategyGenerator(StrategyGenerator):
    """
    MatMulStrategyGenerator is used to generate the sharding strategies when the second tensor is
    a 2D tensor. This is used for nn.Linear, F.linear, torch.matmul and torch.addmm.

    A matmul can be formulated as [n, p] x [p, q] = [n, q]

    Args:
        is_linear (bool): whether this generator is used for nn.Linear and F.linear.
            This will incur extra transformation of the dim partitioning as the weight is transposed.
    """

    def __init__(self, is_linear: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_linear = is_linear

        # as the weight for the linear module is transposed, we can compute
        # the correponding dimension indexfor convenience
        if is_linear:
            self.dim_q = 0
            self.dim_p = 1
        else:
            self.dim_q = 1
            self.dim_p = 0

    def validate(self, input, other, bias) -> bool:
        # make sure the second tensor is a 2D tensor
        assert input.dim() > 0 and other.dim() == 2

        # make sure bias is of the same dimension
        if self.is_linear:
            assert bias is None or bias.shape[-1] == other.shape[0]
        else:
            assert bias is None or bias.shape[-1] == other.shape[1]

    def split_lhs_space_rhs_space(self, mesh_dim_0, mesh_dim_1):
        # handle case SS = SR x RS
        name = f'S{mesh_dim_0}S{mesh_dim_1} = S{mesh_dim_0}R x RS{mesh_dim_1}'

        dim_partition_dict = {
            "input": {
                0: [mesh_dim_0]
            },
            "other": {
                self.dim_q: [mesh_dim_1]
            },
            "bias": {
                -1: [mesh_dim_1]
            },
            "output": {
                0: [mesh_dim_0],
                -1: [mesh_dim_1]
            },
        }
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict)

    def split_lhs_space_both_contract(self, mesh_dim_0, mesh_dim_1):
        # handle the case SR = SS x SR
        name = f'S{mesh_dim_0}R = S{mesh_dim_0}S{mesh_dim_1} x S{mesh_dim_1}R'
        dim_partition_dict = {
            "input": {
                0: [mesh_dim_0],
                -1: [mesh_dim_1]
            },
            "other": {
                self.dim_p: [mesh_dim_1]
            },
            "bias": {},
            "output": {
                0: [mesh_dim_0]
            },
        }
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict, all_reduce_axis=[mesh_dim_1])

    def split_rhs_space_both_contract(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_1} = RS{mesh_dim_0} x S{mesh_dim_0}S{mesh_dim_1}'
        dim_partition_dict = {
            "input": {
                -1: [mesh_dim_0]
            },
            "other": {
                self.dim_p: [mesh_dim_0],
                self.dim_q: [mesh_dim_1]
            },
            "bias": {
                -1: [mesh_dim_1]
            },
            "output": {
                -1: [mesh_dim_1]
            },
        }
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict)

    def recompute_split_both_contract(self, mesh_dim):
        name = f'RR = RS{mesh_dim} x S{mesh_dim}R'
        dim_partition_dict = {
            "input": {
                -1: [mesh_dim]
            },
            "other": {
                self.dim_p: [mesh_dim]
            },
            "bias": {},
            "output": {},
        }
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict, all_reduce_axis=[mesh_dim])

    def split_rhs_space_only(self, mesh_dim):
        name = f'RS{mesh_dim} = RR x RS{mesh_dim}'
        dim_partition_dict = {
            "input": {},
            "other": {
                self.dim_q: [mesh_dim]
            },
            "bias": {
                -1: [mesh_dim]
            },
            "output": {
                -1: [mesh_dim]
            },
        }
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict, all_reduce_axis=[mesh_dim])

    def split_lhs_1st_dim_1d(self, mesh_dim_0, mesh_dim_1):
        name = f'S{mesh_dim_0}{mesh_dim_1}R = S{mesh_dim_0}{mesh_dim_1}R x RR'
        dim_partition_dict = {
            "input": {
                0: [mesh_dim_0, mesh_dim_1]
            },
            "other": {},
            "bias": {},
            "output": {
                0: [mesh_dim_0, mesh_dim_1]
            },
        }
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict)

    def split_lhs_2nd_dim_1d(self, mesh_dim_0, mesh_dim_1):
        name = f'RR = RS{mesh_dim_0}{mesh_dim_1} x S{mesh_dim_0}{mesh_dim_1}R'
        dim_partition_dict = {
            "input": {
                -1: [mesh_dim_0, mesh_dim_1]
            },
            "other": {
                self.dim_p: [mesh_dim_0, mesh_dim_1]
            },
            "bias": {},
            "output": {},
        }
        return IntermediateStrategy(name=name,
                                    dim_partition_dict=dim_partition_dict,
                                    all_reduce_axis=[mesh_dim_0, mesh_dim_1])

    def split_rhs_2nd_dim_1d(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_0}{mesh_dim_1} = RR x RS{mesh_dim_0}{mesh_dim_1}'

        dim_partition_dict = {
            "input": {},
            "other": {
                self.dim_q: [mesh_dim_0, mesh_dim_1]
            },
            "bias": {
                -1: [mesh_dim_0, mesh_dim_1]
            },
            "output": {
                -1: [mesh_dim_0, mesh_dim_1]
            },
        }
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict)


class BatchedMatMulStrategyGenerator(StrategyGenerator):
    """
    Generate sharding strategies for the batched matrix multiplication.

    A batched matrix multiplication can be viewed as
    [b, i, k] x [b, k, j] -> [b, i, j]
    """

    def __init__(self, is_torch_bmm: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_torch_bmm = is_torch_bmm

    def validate(self, input, other, bias) -> bool:
        if self.is_torch_bmm:
            assert input.shape == other.shape
            assert input.dim() > 2
            assert other.shape[-1] == bias.shape[0]
        else:
            # TODO: validate these inputs are broadcastable
            pass

    def split_one_batch_dim(self):
        if 1 in self.device_mesh.mesh_shape:
            mesh_dim = self.device_mesh.mesh_shape.index(1)
            name = f'Sb{mesh_dim} = Sb{mesh_dim} x Sb{mesh_dim}'
            dim_partition_dict = {
                "input": {
                    0: [mesh_dim]
                },
                "other": {
                    0: [mesh_dim]
                },
                "bias": {},
                "output": {
                    0: [mesh_dim]
                }
            }
            return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict)
        else:
            return None

    def split_two_batch_dim(self, mesh_dim_0, mesh_dim_1):
        name = f'Sb{mesh_dim_0}{mesh_dim_1} = Sb{mesh_dim_0}{mesh_dim_1} x Sb{mesh_dim_0}{mesh_dim_1}'
        dim_partition_dict = {
            "input": {
                0: [mesh_dim_0, mesh_dim_1]
            },
            "other": {
                0: [mesh_dim_0, mesh_dim_1]
            },
            "bias": {},
            "output": {
                0: [mesh_dim_0, mesh_dim_1]
            }
        }
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict)

    def split_one_batch_dim(self, mesh_dim):
        name = f'Sb{mesh_dim} = Sb{mesh_dim} x Sb{mesh_dim}'
        dim_partition_dict = {"input": {0: [mesh_dim]}, "other": {0: [mesh_dim]}, "bias": {}, "output": {0: [mesh_dim]}}
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict)

    def split_batch_dim_lhs_space(self, mesh_dim_0, mesh_dim_1):
        name = f'Sb{mesh_dim_0}Si{mesh_dim_1} = Sb{mesh_dim_0}Si{mesh_dim_1} x Sb{mesh_dim_0}'
        dim_partition_dict = {
            "input": {
                0: [mesh_dim_0],
                -2: [mesh_dim_1]
            },
            "other": {
                0: [mesh_dim_0]
            },
            "bias": {},
            "output": {
                0: mesh_dim_0,
                -2: [mesh_dim_1]
            }
        }
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict)

    def split_batch_dim_rhs_space(self, mesh_dim_0, mesh_dim_1):
        name = f'Sb{mesh_dim_0}Sj{mesh_dim_1} = Sb{mesh_dim_0}R x Sb{mesh_dim_0}Sj{mesh_dim_1}'
        dim_partition_dict = {
            "input": {
                0: [mesh_dim_0]
            },
            "other": {
                0: [mesh_dim_0],
                -1: [mesh_dim_1]
            },
            "bias": {
                -1: [mesh_dim_1]
            },
            "output": {
                0: [mesh_dim_0],
                -1: [mesh_dim_1]
            }
        }
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict)

    def split_batch_dim_both_contract(self, mesh_dim_0, mesh_dim_1):
        name = f'Sb{mesh_dim_0}R = Sb{mesh_dim_0}Sk{mesh_dim_1} x Sb{mesh_dim_0}Sk{mesh_dim_1}'
        dim_partition_dict = {
            "input": {
                0: [mesh_dim_0],
                -1: [mesh_dim_1]
            },
            "other": {
                0: [mesh_dim_0],
                -2: [mesh_dim_1]
            },
            "bias": {},
            "output": {
                0: [mesh_dim_0],
                -2: [mesh_dim_1]
            }
        }
        return IntermediateStrategy(name=name, dim_partition_dict=dim_partition_dict, all_reduce_axis=[mesh_dim_1])

    def generate(self) -> List[IntermediateStrategy]:
        strategy_list = []

        # split only the batch dimension
        # Sb = Sb x Sb
        # can be None as it is only for 1D device mesh
        strategy = self.split_one_batch_dim()
        if strategy:
            strategy_list.append(strategy)

        # split batch dim of two inputs and the i dim of the first tensor
        # SbSi = SbSi x Sb
        strategy_list.append(self.split_batch_dim_lhs_space(0, 1))
        strategy_list.append(self.split_batch_dim_lhs_space(1, 0))

        # split batch dim of two inputs and the j of the second tensor
        # SbSj = Sb x SbSj
        strategy_list.append(self.split_batch_dim_rhs_space(0, 1))
        strategy_list.append(self.split_batch_dim_rhs_space(1, 0))

        # split batch dim of two inputs and the k dim of two inputs
        # Sb = SbSk x SbSk, need to all-reduce by k dim
        strategy_list.append(self.split_batch_dim_both_contract(0, 1))
        strategy_list.append(self.split_batch_dim_both_contract(1, 0))

        # split two batch dim
        strategy_list.append(self.split_two_batch_dim(0, 1))
        strategy_list.append(self.split_two_batch_dim(1, 0))

        return strategy_list


class DotHandler(OperatorHandler):
    """
    A OperatorHandler which deals with the sharding strategies for nn.Linear and F.linear.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_data = self.predecessor_node[0]._meta_data
        self.weight = self.module_named_parameters['weight']
        self.output_data = self.node._meta_data

    def _generate_compute_cost(self, input_shape, weight_shape, total_sharding_size):
        # TODO: consider bias addition
        compute_cost = reduce(operator.mul, input_shape) * weight_shape[0] * 2 // total_sharding_size
        return compute_cost

    @ignore_sharding_exception
    def split_lhs_space_rhs_space(self, mesh_dim_0, mesh_dim_1):
        # handle case SS = SR x RS
        name = f'S{mesh_dim_0}S{mesh_dim_1} = S{mesh_dim_0}R x RS{mesh_dim_1}'

        dim_partition_dict_for_input = {0: [mesh_dim_0]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        # linear layer weight is transposed during init
        dim_partition_dict_for_weight = {0: [mesh_dim_1]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_input)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute computation cost
        total_sharding_size = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        compute_cost = self._generate_compute_cost(self.input_data.shape, self.weight.shape, total_sharding_size)

        # compute the memory cost of this strategy
        toatl_memory_cost, activation_memory_cost, weight_memory_cost, input_grad_memory_cost = self._generate_memory_cost(
            dim_partition_dict_for_output, dim_partition_dict_for_weight, dim_partition_dict_for_input)

        # compute the communication cost
        communication_cost_activation_backward = self.device_mesh.all_reduce_cost(activation_memory_cost, mesh_dim_1)
        communication_cost_weight_backward = self.device_mesh.all_reduce_cost(weight_memory_cost, mesh_dim_0)
        communication_cost = communication_cost_activation_backward + communication_cost_weight_backward

        # create and register strategy
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=toatl_memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def split_lhs_space_both_contract(self, mesh_dim_0, mesh_dim_1):
        # handle the case SR = SS x SR
        name = f'S{mesh_dim_0}R = S{mesh_dim_0}S{mesh_dim_1} x S{mesh_dim_1}R'

        dim_partition_dict_for_input = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        # since weight of the linear layer is transposed
        # the actual dim to be sharded is 1
        dim_partition_dict_for_weight = {1: [mesh_dim_1]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {0: [mesh_dim_0]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        total_sharding_size = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        compute_cost = self._generate_compute_cost(self.input_data.shape, self.weight.shape, total_sharding_size)

        # compute the memory cost of this strategy
        toatl_memory_cost, activation_memory_cost, weight_memory_cost, input_grad_memory_cost = self._generate_memory_cost(
            dim_partition_dict_for_output, dim_partition_dict_for_weight, dim_partition_dict_for_input)

        # compute the communication cost of this strategy
        communication_cost_activation_forward = self.device_mesh.all_reduce_cost(activation_memory_cost, mesh_dim_1)
        communication_cost_grad_backward = self.device_mesh.all_reduce_cost(weight_memory_cost, mesh_dim_0)
        communication_cost = communication_cost_activation_forward + communication_cost_grad_backward
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=toatl_memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def split_rhs_space_both_contract(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_1} = RS{mesh_dim_0} x S{mesh_dim_0}S{mesh_dim_1}'

        dim_partition_dict_for_input = {1: [mesh_dim_0]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {1: [mesh_dim_1]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_input)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        total_sharding_size = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        compute_cost = self._generate_compute_cost(self.input_data.shape, self.weight.shape, total_sharding_size)

        # compute the memory cost of this strategy
        toatl_memory_cost, activation_memory_cost, weight_memory_cost, input_grad_memory_cost = self._generate_memory_cost(
            dim_partition_dict_for_output, dim_partition_dict_for_weight, dim_partition_dict_for_input)

        # compute the communication cost of this strategy
        communication_cost_activation_forward = self.device_mesh.all_reduce_cost(activation_memory_cost, mesh_dim_0)
        communication_cost_activation_backward = self.device_mesh.all_reduce_cost(input_grad_memory_cost, mesh_dim_1)
        communication_cost = communication_cost_activation_backward + communication_cost_activation_forward

        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=toatl_memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def recompute_split_both_contract(self, mesh_dim):
        name = f'RR = RS{mesh_dim} x S{mesh_dim}R'

        dim_partition_dict_for_input = {1: [mesh_dim]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {1: [mesh_dim]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        total_sharding_size = self.device_mesh.shape[mesh_dim]
        compute_cost = self._generate_compute_cost(self.input_data.shape, self.weight.shape, total_sharding_size)

        # compute the memory cost of this strategy
        toatl_memory_cost, activation_memory_cost, weight_memory_cost, input_grad_memory_cost = self._generate_memory_cost(
            dim_partition_dict_for_output, dim_partition_dict_for_weight, dim_partition_dict_for_input)

        # compute the communication cost of this strategy
        communication_cost = self.device_mesh.all_reduce_cost(activation_memory_cost, mesh_dim)
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=toatl_memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def split_rhs_space_only(self, mesh_dim):
        name = f'RS{mesh_dim} = RR x RS{mesh_dim}'

        dim_partition_dict_for_input = {}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {0: [mesh_dim]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {1: [mesh_dim]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        total_sharding_size = self.device_mesh.shape[mesh_dim]
        compute_cost = self._generate_compute_cost(self.input_data.shape, self.weight.shape, total_sharding_size)

        # compute the memory cost of this strategy
        toatl_memory_cost, activation_memory_cost, weight_memory_cost, input_grad_memory_cost = self._generate_memory_cost(
            dim_partition_dict_for_output, dim_partition_dict_for_weight, dim_partition_dict_for_input)

        # compute the communication cost of this strategy
        communication_cost_activation_backward = self.device_mesh.all_reduce_cost(input_grad_memory_cost, mesh_dim)
        communication_cost = communication_cost_activation_backward
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=toatl_memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def split_lhs_1st_dim_1d(self, mesh_dim_0, mesh_dim_1):
        name = f'S{mesh_dim_0}{mesh_dim_1}R = S{mesh_dim_0}{mesh_dim_1}R x RR'

        dim_partition_dict_for_input = {0: [mesh_dim_0, mesh_dim_1]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {0: [mesh_dim_0, mesh_dim_1]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        total_sharding_size = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        compute_cost = self._generate_compute_cost(self.input_data.shape, self.weight.shape, total_sharding_size)

        # compute the memory cost of this strategy
        toatl_memory_cost, activation_memory_cost, weight_memory_cost, input_grad_memory_cost = self._generate_memory_cost(
            dim_partition_dict_for_output, dim_partition_dict_for_weight, dim_partition_dict_for_input)

        # compute the communication cost of this strategy
        communication_cost_weight_backward = self.device_mesh.flatten_device_mesh.all_reduce_cost(weight_memory_cost, 0)
        communication_cost = communication_cost_weight_backward
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=toatl_memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def split_lhs_2nd_dim_1d(self, mesh_dim_0, mesh_dim_1):
        name = f'RR = RS{mesh_dim_0}{mesh_dim_1} x S{mesh_dim_0}{mesh_dim_1}R'

        dim_partition_dict_for_input = {1: [mesh_dim_0, mesh_dim_1]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {0: [mesh_dim_0, mesh_dim_1]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        total_sharding_size = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        compute_cost = self._generate_compute_cost(self.input_data.shape, self.weight.shape, total_sharding_size)

        # compute the memory cost of this strategy
        toatl_memory_cost, activation_memory_cost, weight_memory_cost, input_grad_memory_cost = self._generate_memory_cost(
            dim_partition_dict_for_output, dim_partition_dict_for_weight, dim_partition_dict_for_input)

        # compute the communication cost of this strategy
        communication_cost_forward_activation = self.device_mesh.flatten_device_mesh.all_reduce_cost(
            activation_memory_cost, 0)
        communication_cost = communication_cost_forward_activation
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=toatl_memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def split_rhs_2nd_dim_1d(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_0}{mesh_dim_1} = RR x RS{mesh_dim_0}{mesh_dim_1}'

        dim_partition_dict_for_input = {}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {1: [mesh_dim_0, mesh_dim_1]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {1: [mesh_dim_0, mesh_dim_1]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        total_sharding_size = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        compute_cost = self._generate_compute_cost(self.input_data.shape, self.weight.shape, total_sharding_size)

        # compute the memory cost of this strategy
        toatl_memory_cost, activation_memory_cost, weight_memory_cost, input_grad_memory_cost = self._generate_memory_cost(
            dim_partition_dict_for_output, dim_partition_dict_for_weight, dim_partition_dict_for_input)
        # compute the communication cost of this strategy
        communication_cost_activation_backward = self.device_mesh.flatten_device_mesh.all_reduce_cost(
            input_grad_memory_cost, 0)
        communication_cost = communication_cost_activation_backward
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=toatl_memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    def register_strategy(self) -> StrategiesVector:
        '''
        Generate every possible strategies for a linear node, and record all strategies into the strategies_vector.

        Output:

        '''
        # SS = SR x RS
        self.split_lhs_space_rhs_space(0, 1)
        self.split_lhs_space_rhs_space(1, 0)

        # SR = SS x SR
        self.split_lhs_space_both_contract(0, 1)
        self.split_lhs_space_both_contract(1, 0)

        # RS = RS x SS
        self.split_rhs_space_both_contract(0, 1)
        self.split_rhs_space_both_contract(1, 0)

        # RR= RS x SR
        self.recompute_split_both_contract(0)
        self.recompute_split_both_contract(1)

        # RS = RR x RS
        self.split_rhs_space_only(0)
        self.split_rhs_space_only(1)

        # S01R = S01R x RR
        self.split_lhs_1st_dim_1d(0, 1)

        # RR = RS01 x S01R
        self.split_lhs_2nd_dim_1d(0, 1)

        # RS01 = RR x RS01
        self.split_rhs_2nd_dim_1d(0, 1)

        return self.strategies_vector
