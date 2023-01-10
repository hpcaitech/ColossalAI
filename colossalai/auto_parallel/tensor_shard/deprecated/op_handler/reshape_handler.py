import colorsys
import math
import warnings
from copy import deepcopy

import torch

from colossalai.auto_parallel.tensor_shard.deprecated._utils import ignore_sharding_exception
from colossalai.auto_parallel.tensor_shard.deprecated.sharding_strategy import ShardingStrategy, StrategiesVector
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec

from ..constants import INFINITY_COST
from .operator_handler import OperatorHandler


class ReshapeHandler(OperatorHandler):
    """
    An OperatorHandler which deals with the sharding strategies of Reshape Operator, such as torch.reshape, torch.flatten, etc.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_data = self.predecessor_node[0]._meta_data
        self.output_data = self.node._meta_data

    def _generate_compute_cost(self, *args, **kwargs):
        return super()._generate_compute_cost(*args, **kwargs)

    @ignore_sharding_exception
    def register_strategy(self):
        # TODO: add strategies with more output sharding specs other than only fully replicated.
        input_node = self.strategies_vector.predecessor_nodes[0]
        # For reshape function, to keep the computing correctness we keep the sharding
        # spec of input is fully replicated. In addition, we will keep the output in
        # replica status and let the successor node choose the way to resharding the
        # output node. Therefore, the different strategies of input node with same
        # output sharding spec will generate same strategy for reshape function.
        sharding_spec_checklist = []
        for strategy in input_node.strategies_vector:
            # It looks a little bit confusing, the input of the processing node
            # is the output of the input_node.
            input_sharding_spec = strategy.output_sharding_spec
            assert isinstance(input_sharding_spec, ShardingSpec), f'The input node should NOT be a tuple of tensor.'
            if input_sharding_spec in sharding_spec_checklist:
                continue
            sharding_spec_checklist.append(input_sharding_spec)
            dim_partition_dict_for_output = {}
            if isinstance(self.output_data, tuple):
                dim_partition_dict_for_output = [{} for _ in range(len(self.output_data))]
            try:
                if isinstance(self.output_data, tuple):
                    output_sharding_spec = []
                    for output, dim_partition_dict in zip(self.output_data, dim_partition_dict_for_output):
                        output_sharding_spec.append(self._generate_sharding_spec(output, dim_partition_dict))
                else:
                    output_sharding_spec = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)
            except AssertionError as e:
                warnings.warn(f'{e}')
                continue
            name = f'{input_sharding_spec.sharding_sequence} -> FULLY REPLICATED'
            # TODO: use meta_info_prop to profile memory cost and compute cost
            compute_cost = 0
            # consider node._meta_data is in type of tuple
            memory_cost = 0

            # compute the communication cost, in reshape op, the communication happens during casting the input sharding spec to fully replicating.
            dim_partition_dict_for_replicate_input = {}
            replicate_input_sharding_spec = self._generate_sharding_spec(self.input_data,
                                                                         dim_partition_dict_for_replicate_input)
            # shape consistency manager is a singleton class
            shape_consistency_manager = ShapeConsistencyManager()
            _, _, communication_cost = shape_consistency_manager.shape_consistency(input_sharding_spec,
                                                                                   replicate_input_sharding_spec)
            communication_cost = communication_cost["total"]

            # generate resharding cost
            resharding_costs = self._generate_resharding_costs([input_sharding_spec])

            # to prevent the resharding happening, set their resharding cost to inf.
            resharding_costs[input_node] = [0 if cost == 0 else INFINITY_COST for cost in resharding_costs[input_node]]
            sharding_strategy = ShardingStrategy(name,
                                                 output_sharding_spec,
                                                 compute_cost=compute_cost,
                                                 communication_cost=communication_cost,
                                                 memory_cost=memory_cost,
                                                 resharding_costs=resharding_costs,
                                                 input_shardings=[input_sharding_spec])
            self.strategies_vector.append(sharding_strategy)
