import torch
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler.layer_norm_handler import \
    LayerNormModuleHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import (OperationData, OperationDataType, StrategiesVector)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.fx.tracer.meta_patch.patched_module import linear


def test_ln_module_handler():
    model = nn.Sequential(nn.LayerNorm(16).to('meta'))
    tracer = ColoTracer()
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     %_0 : [#users=1] = call_module[target=0](args = (%input_1,), kwargs = {})
    #     return _0
    graph = tracer.trace(model, meta_args={"input": torch.rand(4, 16).to('meta')})
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    ln_mod_node = list(graph.nodes)[1]
    strategies_vector = StrategiesVector(ln_mod_node)

    # build handler
    handler = LayerNormModuleHandler(node=ln_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping['input'].name == "input_1"
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([4, 16])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([4, 16])

    assert mapping['other'].name == "weight"
    assert mapping['other'].data.is_meta
    assert mapping['other'].data.shape == torch.Size([16])
    assert mapping['other'].type == OperationDataType.PARAM
    assert mapping['other'].logical_shape == torch.Size([16])

    assert mapping['bias'].name == "bias"
    assert mapping['bias'].data.is_meta
    assert mapping['bias'].data.shape == torch.Size([16])
    assert mapping['bias'].type == OperationDataType.PARAM
    assert mapping['bias'].logical_shape == torch.Size([16])

    assert mapping['output'].name == "_0"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([4, 16])
    assert mapping['output'].type == OperationDataType.OUTPUT

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # SR = SR x R
    assert '[S0, R] = [S0, R] x [R]' in strategy_name_list
    assert '[S1, R] = [S1, R] x [R]' in strategy_name_list

    # RR = RR x R
    assert 'RR = RR x R' in strategy_name_list

    # S01R = S01R x R
    assert '[S01, R] = [S01, R] x [R]' in strategy_name_list


if __name__ == '__main__':
    test_ln_module_handler()
