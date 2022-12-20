import pytest
import torch
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler.normal_pooling_handler import NormPoolingHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.fx.tracer.meta_patch.patched_module import linear
from colossalai.testing.pytest_wrapper import run_on_environment_flag


@run_on_environment_flag(name='AUTO_PARALLEL')
def test_norm_pool_handler():
    model = nn.Sequential(nn.MaxPool2d(4, padding=1).to('meta'))
    tracer = ColoTracer()
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     %_0 : [#users=1] = call_module[target=0](args = (%input_1,), kwargs = {})
    #     return _0
    graph = tracer.trace(model, meta_args={"input": torch.rand(4, 4, 64, 64).to('meta')})

    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    conv_mod_node = list(graph.nodes)[1]
    strategies_vector = StrategiesVector(conv_mod_node)

    # build handler
    handler = NormPoolingHandler(node=conv_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)
    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.data is not None

    assert mapping['input'].name == "input_1"
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([4, 4, 64, 64])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([4, 4, 64, 64])

    assert mapping['output'].name == "_0"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([4, 4, 16, 16])
    assert mapping['output'].type == OperationDataType.OUTPUT

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]
    assert len(strategy_name_list) == 9


if __name__ == '__main__':
    test_norm_pool_handler()
