import torch
from torch.fx import GraphModule
import torch.nn as nn
import pytest

from colossalai.fx.proxy import ColoProxy
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.tensor.sharding_spec import ShardingSpec, _DimSpec
from colossalai.auto_parallel.tensor_shard.deprecated.op_handler.batch_norm_handler import BatchNormHandler
from colossalai.auto_parallel.tensor_shard.deprecated.sharding_strategy import ShardingStrategy, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh


class BNModel(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        x = x * 2
        x = self.bn(x)
        return x


def test_bn_handler():
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    entire_shape = torch.Size((4, 16, 64, 64))

    tracer = ColoTracer()
    model = BNModel(16)
    input_sample = {'x': torch.rand(4, 16, 64, 64).to('meta')}
    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %mul : [#users=1] = call_function[target=operator.mul](args = (%x, 2), kwargs = {})
    #     %bn : [#users=1] = call_module[target=bn](args = (%mul,), kwargs = {})
    #     return bn
    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()
    # [x, mul, bn, output]
    nodes = [node for node in gm.graph.nodes]

    # find the sharding strategies for the input node of the bn node
    # strategies_for_input = [[R, R, R, R], [R, S0, R, R], [R, S1, R, R], [S0, R, R, R], [S0, S1, R, R], [S1, R, R, R], [S1, S0, R, R]]
    strategies_vector_for_input = StrategiesVector(nodes[1])
    sharding_option = (None, 0, 1)
    for first_sharding_index in sharding_option:
        for second_sharding_index in sharding_option:
            if first_sharding_index is not None and second_sharding_index == first_sharding_index:
                continue
            if first_sharding_index is None:
                first_dim_spec = _DimSpec([])
            else:
                first_dim_spec = _DimSpec([first_sharding_index])

            if second_sharding_index is None:
                second_dim_spec = _DimSpec([])
            else:
                second_dim_spec = _DimSpec([second_sharding_index])

            replica_dim_spec = _DimSpec([])
            sharding_sequence = [first_dim_spec, second_dim_spec, replica_dim_spec, replica_dim_spec]
            sharding_spec = ShardingSpec(device_mesh=device_mesh,
                                         entire_shape=entire_shape,
                                         sharding_sequence=sharding_sequence)
            strategy_name = str(sharding_spec.sharding_sequence)
            sharding_strategy = ShardingStrategy(name=strategy_name, output_sharding_spec=sharding_spec)
            strategies_vector_for_input.append(sharding_strategy)
    setattr(nodes[1], 'strategies_vector', strategies_vector_for_input)

    # generate bn strategy
    strategies_vector = StrategiesVector(node=nodes[2])
    bn_handler = BatchNormHandler(
        node=nodes[2],
        device_mesh=device_mesh,
        strategies_vector=strategies_vector,
    )
    bn_handler.register_strategy()
    # ['RS0 = RS0 x S0', 'S1S0 = RS0 x S0', 'RS1 = RS1 x S1', 'S0S1 = RS1 x S1', 'RR = RR x R', 'S0R = RR x R', 'S1R = RR x R', 'S01R = RR x R', 'RS01 = RS01 x S01',
    # 'S0R = S0R x R WITH SYNC_BN', 'S1R = S1R x R WITH SYNC_BN', 'S0S1 = S0S1 x S1 WITH SYNC_BN', 'S1S0 = S1S0 x S0 WITH SYNC_BN', 'S01R = S01R x R WITH SYNC_BN']
    strategy_name_list = [strategy.name for strategy in bn_handler.strategies_vector]

    # RS = RS x S and strategies based on it, such as
    # SS = RS x S
    assert 'RS0 = RS0 x S0' in strategy_name_list
    assert 'S1S0 = RS0 x S0' in strategy_name_list
    assert 'RS1 = RS1 x S1' in strategy_name_list
    assert 'S0S1 = RS1 x S1' in strategy_name_list

    # RR = RR x R and strategies based on it, such as
    # SR = SR x R
    assert 'RR = RR x R' in strategy_name_list
    assert 'S0R = RR x R' in strategy_name_list
    assert 'S1R = RR x R' in strategy_name_list
    assert 'S01R = RR x R' in strategy_name_list

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
    test_bn_handler()
