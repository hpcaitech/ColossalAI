import torch
import torch.nn as nn
import transformers
from torch.fx import GraphModule

from colossalai.auto_parallel.tensor_shard.constants import BATCHNORM_MODULE_OP
from colossalai.auto_parallel.tensor_shard.solver import (
    CostGraph,
    GraphAnalyser,
    Solver,
    SolverOptions,
    StrategiesConstructor,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.testing import parameterize
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from tests.test_auto_parallel.test_tensor_shard.test_gpt.gpt_modules import GPT2MLP, GPT2Attention, GPT2Block, GPT2Model

BATCH_SIZE = 1
SEQ_LENGTH = 32
HIDDEN_DIM = 768


@run_on_environment_flag(name='AUTO_PARALLEL')
@parameterize('model_cls', [GPT2Block, GPT2Attention, GPT2MLP, GPT2Model])
def test_self_attention_block(model_cls):
    config = transformers.GPT2Config(n_position=64, n_layer=4, n_head=16, n_embd=HIDDEN_DIM)
    if model_cls == GPT2MLP:
        model = model_cls(intermediate_size=4 * config.hidden_size, config=config)
    else:
        model = model_cls(config=config)
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    shape_consistency_manager = ShapeConsistencyManager()

    tracer = ColoTracer()
    if model_cls == GPT2MLP:
        input_sample = {
            'hidden_states': torch.rand(BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM).to('meta'),
        }
    elif model_cls in (GPT2Attention, GPT2Block):
        input_sample = {
            'hidden_states': torch.rand(BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM).to('meta'),
            'attention_mask': torch.rand(1, SEQ_LENGTH).to('meta'),
        }
    else:
        input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
        attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        input_sample = {k: v.to('meta') for k, v in kwargs.items()}

    graph = tracer.trace(root=model, meta_args=input_sample)

    gm = GraphModule(model, graph, model.__class__.__name__)
    print(gm.graph)
    gm.recompile()
    graph_analyser = GraphAnalyser(gm)
    liveness_list = graph_analyser.liveness_analysis()
    solver_options = SolverOptions()
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)
    strategies_constructor.build_strategies_and_cost()

    cost_graph = CostGraph(strategies_constructor.leaf_strategies)
    cost_graph.simplify_graph()
    solver = Solver(gm.graph, strategies_constructor, cost_graph, graph_analyser, memory_budget=-1)
    ret = solver.call_solver_serialized_args()
    strategies_list = solver.last_s_val
    nodes = [strategies_vector.node for strategies_vector in strategies_constructor.leaf_strategies]

    computation_cost = 0
    communication_cost = 0
    memory_cost = 0
    for index, node in enumerate(nodes):
        print(node.name, node.strategies_vector[strategies_list[index]].name)
        computation_cost += node.strategies_vector[strategies_list[index]].compute_cost.total
        communication_cost += node.strategies_vector[strategies_list[index]].communication_cost.total
        node_memory_cost = node.strategies_vector[strategies_list[index]].memory_cost.total
        if isinstance(node_memory_cost, tuple):
            node_memory_cost = node_memory_cost[0]
        memory_cost += node_memory_cost.activation + node_memory_cost.parameter

    print(f'computation cost is {computation_cost}')
    print(f'communication cost is {communication_cost}')
    print(f'memory cost is {memory_cost}')


if __name__ == '__main__':
    test_self_attention_block()
