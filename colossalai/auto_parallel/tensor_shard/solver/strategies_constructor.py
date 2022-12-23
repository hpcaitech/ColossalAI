import builtins
import math
import operator
from copy import deepcopy
from typing import Dict, List

import torch
from torch.fx import Graph, Node

from colossalai.auto_parallel.tensor_shard.node_handler import (
    GetattrHandler,
    OutputHandler,
    PlaceholderHandler,
    operator_registry,
)
from colossalai.auto_parallel.tensor_shard.sharding_strategy import StrategiesVector
from colossalai.auto_parallel.tensor_shard.utils import generate_resharding_costs, generate_sharding_spec
from colossalai.device.device_mesh import DeviceMesh

from .options import DataloaderOption, SolverOptions

__all__ = ['StrategiesConstructor']


class StrategiesConstructor:
    """
    StrategiesConstructor is used to construct the parallelization plan for the model execution.

    Args:
        graph (Graph): a Graph object used for analysis and strategy generation.
        device_mesh (DeviceMesh): a DeviceMesh object which contains the meta information about the cluster.
        solver_options (SolverOptions): a SolverOptions object which specifies the preferences for plan searching.
    """

    def __init__(self, graph: Graph, device_mesh: DeviceMesh, solver_options: SolverOptions):
        self.graph = graph
        assert graph.owning_module is not None, 'The given graph is not associated with a owning_module'
        self.root_module = self.graph.owning_module
        self.nodes = list(graph.nodes)
        self.device_mesh = device_mesh
        self.leaf_strategies = []
        self.strategy_map = {}
        self.solver_options = solver_options
        self.no_strategy_nodes = []

    def remove_duplicated_strategy(self, strategies_vector):
        '''
        In build_strategies_and_cost method, we may produce some duplicated strategies.
        In this method, we will remove the duplicated strategies depending on the strategies name.
        Note that this operation is in-place.
        '''
        name_checklist = []
        remove_list = []
        for strategy in strategies_vector:
            if strategy.name not in name_checklist:
                name_checklist.append(strategy.name)
            else:
                remove_list.append(strategy)
        for strategy in remove_list:
            strategies_vector.remove(strategy)

    def build_strategies_and_cost(self):
        """
        This method is to build the strategy vector for each node in the computation graph.
        """

        def _check_no_strategy_for_node(node):
            if node.op in ('placeholder', 'get_attr', 'output'):
                return False

            def _check_no_strategy_for_data(data):
                label = True
                if isinstance(data, torch.Tensor):
                    return False
                elif isinstance(data, (tuple, list)):
                    for d in data:
                        label = label and _check_no_strategy_for_data(d)
                return label

            return _check_no_strategy_for_data(node._meta_data)

        for node in self.nodes:
            strategies_vector = StrategiesVector(node)

            if _check_no_strategy_for_node(node):
                self.no_strategy_nodes.append(node)
                pass

            # placeholder node
            elif node.op == 'placeholder':
                if self.solver_options.dataloader_option == DataloaderOption.DISTRIBUTED:
                    placeholder_option = 'distributed'
                else:
                    assert self.solver_options.dataloader_option == DataloaderOption.REPLICATED, f'placeholder_option {self.solver_options.dataloader_option} is not supported'
                    placeholder_option = 'replicated'
                placeholder_handler = PlaceholderHandler(node,
                                                         self.device_mesh,
                                                         strategies_vector,
                                                         placeholder_option=placeholder_option)
                placeholder_handler.register_strategy()

            # get_attr node
            elif node.op == 'get_attr':
                getattr_handler = GetattrHandler(node, self.device_mesh, strategies_vector)
                getattr_handler.register_strategy()

            # call_module node
            elif node.op == 'call_module':
                target = node.target
                submod = self.root_module.get_submodule(target)
                submod_type = type(submod)
                handler = operator_registry.get(submod_type)(node, self.device_mesh, strategies_vector)
                handler.register_strategy()
                # attach metainfo_vector to node
                if hasattr(handler, 'metainfo_vector'):
                    setattr(node, 'metainfo_vector', handler.metainfo_vector)

            # call_function node
            elif node.op == 'call_function':
                target = node.target
                handler = operator_registry.get(target)(node, self.device_mesh, strategies_vector)
                handler.register_strategy()
                # attach metainfo_vector to node
                if hasattr(handler, 'metainfo_vector'):
                    setattr(node, 'metainfo_vector', handler.metainfo_vector)

            # call_method node
            elif node.op == 'call_method':
                method = getattr(node.args[0]._meta_data.__class__, node.target)
                handler = operator_registry.get(method)(node, self.device_mesh, strategies_vector)
                handler.register_strategy()
                # attach metainfo_vector to node
                if hasattr(handler, 'metainfo_vector'):
                    setattr(node, 'metainfo_vector', handler.metainfo_vector)

            # output node
            elif node.op == 'output':
                if self.solver_options.dataloader_option == DataloaderOption.DISTRIBUTED:
                    output_option = 'distributed'
                else:
                    assert self.solver_options.dataloader_option == DataloaderOption.REPLICATED, f'placeholder_option {self.solver_options.dataloader_option} is not supported'
                    output_option = 'replicated'
                output_handler = OutputHandler(node, self.device_mesh, strategies_vector, output_option=output_option)
                output_handler.register_strategy()

            self.remove_duplicated_strategy(strategies_vector)
            setattr(node, 'strategies_vector', strategies_vector)
            self.leaf_strategies.append(strategies_vector)
            self.strategy_map[node] = strategies_vector

        # remove no strategy nodes
        remove_list = []
        for strategies_vector in self.leaf_strategies:
            if len(strategies_vector) == 0:
                remove_list.append(strategies_vector.node)

        for node in remove_list:
            if node.strategies_vector in self.leaf_strategies:
                self.leaf_strategies.remove(node.strategies_vector)
            if node in self.strategy_map:
                self.strategy_map.pop(node)
