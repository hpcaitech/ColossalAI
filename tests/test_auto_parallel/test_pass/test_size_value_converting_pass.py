import pytest
import torch

from colossalai._analyzer.fx.graph_module import ColoGraphModule
from colossalai._analyzer.fx.passes import shape_prop_pass
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.passes.runtime_preparation_pass import size_value_converting_pass
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.sharding_spec import ShardingSpec
from colossalai.testing import clear_cache_before_run


class TestModule(torch.nn.Module):
    def forward(self, x):
        size = x.size()
        return size


def insert_narrow(gm, x_node):
    graph = gm.graph
    with graph.inserting_after(x_node):
        shard_node = graph.create_node("call_method", "narrow", args=(x_node, 0, 0, 2), kwargs={})
    size_node = list(x_node.users.keys())[0]
    size_node.args = (shard_node,)
    return gm


def recover_narrow(gm, narrow_node):
    graph = gm.graph
    size_node = list(graph.nodes)[2]
    x_node = narrow_node.args[0]
    size_node.args = (x_node,)
    graph.erase_node(narrow_node)
    return gm


@pytest.mark.skip("ShapeProp is not compatible with PyTorch 1.11.0")
@clear_cache_before_run()
def test_size_value_converting_pass():
    model = TestModule()
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    meta_args = {"x": torch.rand(4, 8).to("meta")}
    input = torch.rand(4, 8)
    tracer = ColoTracer(bias_addition_split=True)
    graph = tracer.trace(root=model, meta_args=meta_args)
    x_node = list(graph.nodes)[0]
    x_sharding_spec = ShardingSpec(device_mesh, entire_shape=(4, 8), dim_partition_dict={0: [0]})
    setattr(x_node, "sharding_spec", x_sharding_spec)
    gm = ColoGraphModule(model, graph)
    gm = insert_narrow(gm, x_node)
    shape_prop_pass(gm, *meta_args.values())
    gm.recompile()
    size = gm(input)
    assert size == torch.Size([2, 8])

    narrow_node = list(gm.graph.nodes)[1]
    gm = recover_narrow(gm, narrow_node)
    gm = size_value_converting_pass(gm, device_mesh)
    gm = insert_narrow(gm, x_node)
    gm.recompile()
    size = gm(input)
    assert size == torch.Size([4, 8])


if __name__ == "__main__":
    test_size_value_converting_pass()
