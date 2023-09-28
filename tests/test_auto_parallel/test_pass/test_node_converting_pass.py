import torch

from colossalai.auto_parallel.passes.runtime_preparation_pass import node_args_converting_pass
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.tracer import ColoTracer
from colossalai.tensor.sharding_spec import ShardingSpec
from colossalai.testing import clear_cache_before_run


class TestModule(torch.nn.Module):
    def forward(self, x):
        x = x.view(4, 4, 2)
        return x


def insert_narrow(gm, x_node):
    graph = gm.graph
    with graph.inserting_after(x_node):
        shard_node = graph.create_node("call_method", "narrow", args=(x_node, 0, 0, 2), kwargs={})
    view_node = list(x_node.users.keys())[0]
    new_args = list(view_node.args)
    new_args[0] = shard_node
    view_node.args = tuple(new_args)
    return gm


@clear_cache_before_run()
def test_node_args_converting_pass():
    model = TestModule()
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    meta_args = {"x": torch.rand(4, 8).to("meta")}
    input = torch.rand(4, 8)
    tracer = ColoTracer()
    graph = tracer.trace(root=model, meta_args=meta_args)

    x_node = list(graph.nodes)[0]
    view_node = list(graph.nodes)[1]
    sharding_spec = ShardingSpec(device_mesh, entire_shape=(4, 8), dim_partition_dict={0: [0]})
    setattr(x_node, "sharding_spec", sharding_spec)
    setattr(view_node, "sharding_spec", sharding_spec)

    gm = ColoGraphModule(model, graph)
    gm = node_args_converting_pass(gm, device_mesh)
    gm = insert_narrow(gm, x_node)
    gm.recompile()
    output = gm(input)
    assert output.shape == torch.Size([2, 4, 2])


if __name__ == "__main__":
    test_node_args_converting_pass()
