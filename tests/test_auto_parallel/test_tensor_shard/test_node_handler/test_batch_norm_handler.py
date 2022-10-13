from colossalai.fx.tracer.meta_patch.patched_module import linear
import torch
import torch.nn as nn
from colossalai.fx import ColoTracer, ColoGraphModule
from colossalai.auto_parallel.solver.node_handler.batch_norm_handler import BatchNormModuleHandler
from colossalai.auto_parallel.solver.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh


def test_bn_module_handler():
    model = nn.Sequential(nn.BatchNorm2d(16).to('meta'))
    tracer = ColoTracer()
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     %_0 : [#users=1] = call_module[target=0](args = (%input_1,), kwargs = {})
    #     return _0
    graph = tracer.trace(model, meta_args={"input": torch.rand(4, 16, 64, 64).to('meta')})
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    bn_mod_node = list(graph.nodes)[1]
    strategies_vector = StrategiesVector(bn_mod_node)

    # build handler
    handler = BatchNormModuleHandler(node=bn_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping['input'].name == "input_1"
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([4, 16, 64, 64])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([4, 16, 64, 64])

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
    assert mapping['output'].data.shape == torch.Size([4, 16, 64, 64])
    assert mapping['output'].type == OperationDataType.OUTPUT

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # RS = RS x S
    assert 'RS0 = RS0 x S0' in strategy_name_list
    assert 'RS1 = RS1 x S1' in strategy_name_list

    # RR = RR x R
    assert 'RR = RR x R' in strategy_name_list

    # RS01 = RS01 x S01
    assert 'RS01 = RS01 x S01' in strategy_name_list

    # SR = SR x R WITH SYNC_BN
    assert 'S0R = S0R x R WITH SYNC_BN' in strategy_name_list
    assert 'S1R = S1R x R WITH SYNC_BN' in strategy_name_list

    # SS = SS x S WITH SYNC_BN
    assert 'S0S1 = S0S1 x S1 WITH SYNC_BN' in strategy_name_list
    assert 'S1S0 = S1S0 x S0 WITH SYNC_BN' in strategy_name_list

    # S01R = S01R x R WITH SYNC_BN
    assert 'S01R = S01R x R WITH SYNC_BN' in strategy_name_list


if __name__ == '__main__':
    test_bn_module_handler()
