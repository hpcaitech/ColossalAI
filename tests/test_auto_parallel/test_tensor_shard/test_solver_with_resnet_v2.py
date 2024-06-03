import torch
from torch.fx import GraphModule
from torchvision.models import resnet50

from colossalai._analyzer.fx.passes import shape_prop_pass

# from colossalai.fx.tracer.tracer import ColoTracer
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.tensor_shard.constants import BATCHNORM_MODULE_OP
from colossalai.auto_parallel.tensor_shard.options import SolverOptions
from colossalai.auto_parallel.tensor_shard.solver import CostGraph, Solver, StrategiesConstructor
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.testing import clear_cache_before_run, run_on_environment_flag


@run_on_environment_flag(name="AUTO_PARALLEL")
@clear_cache_before_run()
def test_cost_graph():
    physical_mesh_id = torch.arange(0, 8)
    mesh_shape = (2, 4)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    ShapeConsistencyManager()

    tracer = ColoTracer(bias_addition_split=True)
    model = resnet50(num_classes=100000)
    input_sample = {"x": torch.rand(128, 3, 224, 224).to("meta")}

    graph = tracer.trace(root=model, meta_args=input_sample)
    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %conv1 : [#users=1] = call_module[target=conv1](args = (%x,), kwargs = {})
    #     %bn1 : [#users=1] = call_module[target=bn1](args = (%conv1,), kwargs = {})
    #     %relu : [#users=1] = call_module[target=relu](args = (%bn1,), kwargs = {})
    #     %maxpool : [#users=2] = call_module[target=maxpool](args = (%relu,), kwargs = {})
    #     %layer1_0_conv1 : [#users=1] = call_module[target=layer1.0.conv1](args = (%maxpool,), kwargs = {})
    #     %layer1_0_bn1 : [#users=1] = call_module[target=layer1.0.bn1](args = (%layer1_0_conv1,), kwargs = {})
    #     %layer1_0_relu : [#users=1] = call_module[target=layer1.0.relu](args = (%layer1_0_bn1,), kwargs = {})
    #     %layer1_0_conv2 : [#users=1] = call_module[target=layer1.0.conv2](args = (%layer1_0_relu,), kwargs = {})
    #     %layer1_0_bn2 : [#users=1] = call_module[target=layer1.0.bn2](args = (%layer1_0_conv2,), kwargs = {})
    #     %add : [#users=1] = call_function[target=operator.add](args = (%layer1_0_bn2, %maxpool), kwargs = {})
    #     %layer1_0_relu_1 : [#users=2] = call_module[target=layer1.0.relu](args = (%add,), kwargs = {})
    #     %layer1_1_conv1 : [#users=1] = call_module[target=layer1.1.conv1](args = (%layer1_0_relu_1,), kwargs = {})
    #     %layer1_1_bn1 : [#users=1] = call_module[target=layer1.1.bn1](args = (%layer1_1_conv1,), kwargs = {})
    #     %layer1_1_relu : [#users=1] = call_module[target=layer1.1.relu](args = (%layer1_1_bn1,), kwargs = {})
    #     %layer1_1_conv2 : [#users=1] = call_module[target=layer1.1.conv2](args = (%layer1_1_relu,), kwargs = {})
    #     %layer1_1_bn2 : [#users=1] = call_module[target=layer1.1.bn2](args = (%layer1_1_conv2,), kwargs = {})
    #     %add_1 : [#users=1] = call_function[target=operator.add](args = (%layer1_1_bn2, %layer1_0_relu_1), kwargs = {})
    #     ...
    #     %avgpool : [#users=1] = call_module[target=avgpool](args = (%layer4_2_relu_1,), kwargs = {})
    #     %flatten : [#users=1] = call_function[target=torch.flatten](args = (%avgpool, 1), kwargs = {})
    #     %fc : [#users=1] = call_module[target=fc](args = (%flatten,), kwargs = {})
    #     return fc
    gm = GraphModule(model, graph, model.__class__.__name__)
    shape_prop_pass(gm, *input_sample.values())
    gm.recompile()

    solver_options = SolverOptions()
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)
    strategies_constructor.build_strategies_and_cost()

    cost_graph = CostGraph(strategies_constructor.leaf_strategies)
    cost_graph.simplify_graph()
    solver = Solver(gm.graph, strategies_constructor, cost_graph)

    ret = solver.call_solver_serialized_args()
    print(ret[0])
    print(solver.last_s_val)
    strategies_list = solver.last_s_val

    computation_cost = 0
    communication_cost = 0
    communication_cost_bn = 0
    memory_cost = 0
    for index, node in enumerate(graph.nodes):
        if node.op == "call_module":
            submod = node.graph.owning_module.get_submodule(node.target)
            if type(submod) in BATCHNORM_MODULE_OP:
                communication_cost_bn += node.strategies_vector[strategies_list[index]].communication_cost.total
        print(node.name, node.strategies_vector[strategies_list[index]].name)
        computation_cost += node.strategies_vector[strategies_list[index]].compute_cost.total
        communication_cost += node.strategies_vector[strategies_list[index]].communication_cost.total
        node_memory_cost = node.strategies_vector[strategies_list[index]].memory_cost.total
        if isinstance(node_memory_cost, tuple):
            node_memory_cost = node_memory_cost[0]
        memory_cost += node_memory_cost.activation + node_memory_cost.parameter

    print(f"computation cost is {computation_cost}")
    print(f"communication cost is {communication_cost}")
    print(f"memory cost is {memory_cost}")
    print(f"bn communication cost is {communication_cost_bn}")


if __name__ == "__main__":
    test_cost_graph()
