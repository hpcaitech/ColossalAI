import math
import operator
import warnings
from copy import deepcopy
from functools import reduce
from typing import Dict, List

import torch
from colossalai.auto_parallel.tensor_shard.deprecated._utils import \
    ignore_sharding_exception
from colossalai.auto_parallel.tensor_shard.deprecated.constants import \
    INFINITY_COST
from colossalai.auto_parallel.tensor_shard.deprecated.sharding_strategy import (ShardingStrategy, StrategiesVector)
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec

from .operator_handler import OperatorHandler

__all__ = ['UnaryElementwiseHandler']


class UnaryElementwiseHandler(OperatorHandler):
    """
    An OperatorHandler which deals with the sharding strategies of UnaryElementwiseOp.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.node.op == 'call_module':
            target = self.node.target
            submod = self.node.graph.owning_module.get_submodule(target)
            submod_type = type(submod)
            if submod_type == torch.nn.Dropout:
                print(f'predecessor nodes of dropout node are {self.predecessor_node}')
        input_nodes_len = 0
        for check_node in self.predecessor_node:
            if isinstance(check_node._meta_data, torch.Tensor):
                input_nodes_len += 1
        assert input_nodes_len == 1, f'Temporally, we just support single input element-wise op, node name is {self.node}, node args is {self.node.args}.'
        self.input_data = self.predecessor_node[0]._meta_data
        self.input_node = self.predecessor_node[0]
        self.output_data = self.node._meta_data

    def _generate_compute_cost(self, *args, **kwargs):
        return super()._generate_compute_cost(*args, **kwargs)

    @ignore_sharding_exception
    def register_strategy(self):
        # TODO: integrate element-wise func and module together
        # create sharding strategy for element-wise function

        # For element-wise function, we keep the sharding spec of output node same as
        # the input. Therefore, the different strategies of input node with same
        # output sharding spec will generate same strategy for element-wise function.

        for index, strategy in enumerate(self.input_node.strategies_vector):
            # It looks a little bit confusing, the input of the processing node
            # is the output of the input_node.
            input_sharding_spec = strategy.output_sharding_spec
            assert isinstance(input_sharding_spec, ShardingSpec), f'The input node should NOT be a tuple of tensor.'

            dim_partition_dict = deepcopy(input_sharding_spec.dim_partition_dict)
            try:
                output_sharding_spec = self._generate_sharding_spec(self.output_data, dim_partition_dict)
            except AssertionError as e:
                warnings.warn(f'{e}')
                continue
            # add index into name to pass the duplicated check
            # we keep same strategies with different name for node merging, and it will not increase the searching space,
            # because in solver, this node will be merged into other nodes, and solver will not create a new variable for this node.
            name = f'{input_sharding_spec.sharding_sequence} -> {output_sharding_spec.sharding_sequence}_{index}'
            # TODO: use meta_info_prop to profile memory cost and compute cost
            compute_cost = self.output_data.numel()
            memory_cost = 0

            resharding_costs = self._generate_resharding_costs([input_sharding_spec])

            # to prevent the resharding happening, set their resharding cost to inf.
            resharding_costs[self.input_node] = [
                0 if cost == 0 else INFINITY_COST for cost in resharding_costs[self.input_node]
            ]
            sharding_strategy = ShardingStrategy(name,
                                                 output_sharding_spec,
                                                 compute_cost=compute_cost,
                                                 memory_cost=memory_cost,
                                                 resharding_costs=resharding_costs,
                                                 input_shardings=[input_sharding_spec])
            self.strategies_vector.append(sharding_strategy)
