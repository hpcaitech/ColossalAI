import torch
from torch.fx import GraphModule
import torch.nn as nn
import pytest

from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.tensor_shard.deprecated.sharding_strategy import ShardingStrategy, StrategiesVector
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.device.device_mesh import DeviceMesh
from colossalai.auto_parallel.tensor_shard.deprecated.strategies_constructor import StrategiesConstructor
from colossalai.auto_parallel.tensor_shard.deprecated.cost_graph import CostGraph
from copy import deepcopy
from colossalai.auto_parallel.tensor_shard.deprecated import Solver
import transformers
from colossalai.auto_parallel.tensor_shard.deprecated.constants import *
from colossalai.auto_parallel.tensor_shard.deprecated.graph_analysis import GraphAnalyser
from colossalai.auto_parallel.tensor_shard.deprecated.options import SolverOptions
from colossalai.testing.pytest_wrapper import run_on_environment_flag

BATCH_SIZE = 8
SEQ_LENGHT = 8


@run_on_environment_flag(name='AUTO_PARALLEL')
def test_cost_graph():
    physical_mesh_id = torch.arange(0, 8)
    mesh_shape = (2, 4)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    shape_consistency_manager = ShapeConsistencyManager()

    tracer = ColoTracer()
    config = transformers.GPT2Config(n_position=1024, n_layer=1, n_head=12)
    model = transformers.GPT2LMHeadModel(config=config)
    input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
    token_type_ids = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
    attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
    kwargs = dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    meta_args = {k: v.to('meta') for k, v in kwargs.items()}

    graph = tracer.trace(root=model, meta_args=meta_args)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()
    graph_analyser = GraphAnalyser(gm)
    liveness_list = graph_analyser.liveness_analysis()
    solver_options = SolverOptions(fast=True)
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)
    print(graph)
    strategies_constructor.build_strategies_and_cost()
    for check_node, strategies_vector in strategies_constructor.strategy_map.items():
        print(check_node, len(strategies_vector))
    cost_graph = CostGraph(strategies_constructor.leaf_strategies)
    cost_graph.simplify_graph()
    # solver = Solver(gm.graph, strategies_constructor, cost_graph, graph_analyser, memory_budget=1620017824.0)
    solver = Solver(gm.graph, strategies_constructor, cost_graph, graph_analyser)

    ret = solver.call_solver_serialized_args()
    print(ret)
    strategies_list = list(ret[0])
    print(strategies_list)
    computation_cost = 0
    communication_cost = 0
    memory_cost = 0
    nodes = [strategies_vector.node for strategies_vector in strategies_constructor.leaf_strategies]
    for index, node in enumerate(nodes):
        print(node.name, node.strategies_vector[strategies_list[index]].name)
        computation_cost += node.strategies_vector[strategies_list[index]].compute_cost
        communication_cost += node.strategies_vector[strategies_list[index]].communication_cost
        node_memory_cost = node.strategies_vector[strategies_list[index]].memory_cost
        if isinstance(node_memory_cost, tuple):
            node_memory_cost = node_memory_cost[0]
        memory_cost += node_memory_cost

    print(f'computation cost is {computation_cost}')
    print(f'communication cost is {communication_cost}')
    print(f'memory cost is {memory_cost}')


if __name__ == '__main__':
    test_cost_graph()
