from functools import partial

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler import LinearFunctionHandler
from colossalai.auto_parallel.tensor_shard.node_handler.option import ShardOption
from colossalai.auto_parallel.tensor_shard.sharding_strategy import StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.testing import parameterize
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.testing.utils import parameterize


class LinearModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, others, bias=None):
        x = nn.functional.linear(input, others, bias=bias)
        return x


def check_shard_option(shard_option):
    model = LinearModel().cuda()
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)

    tracer = ColoTracer()
    graph = tracer.trace(model,
                         meta_args={
                             "input": torch.rand(4, 4, 4, 16).to('meta'),
                             'others': torch.rand(32, 16).to('meta')
                         })
    gm = ColoGraphModule(model, graph)
    linear_func_node = list(graph.nodes)[2]
    strategies_vector = StrategiesVector(linear_func_node)

    # build handler
    handler = LinearFunctionHandler(node=linear_func_node,
                                    device_mesh=device_mesh,
                                    strategies_vector=strategies_vector,
                                    shard_option=shard_option)

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # SS = SR x RS
    assert 'S1S0 = S1R x RS0_0' in strategy_name_list
    assert 'S0S1 = S0R x RS1_1' in strategy_name_list
    assert 'S0S1 = S0R x RS1_2' in strategy_name_list
    assert 'S0S1 = S0R x RS1_0' in strategy_name_list
    assert 'S1S0 = S1R x RS0_1' in strategy_name_list
    assert 'S1S0 = S1R x RS0_2' in strategy_name_list

    # SR = SS x SR
    assert 'S0R = S0S1 x S1R_1' in strategy_name_list
    assert 'S0R = S0S1 x S1R_2' in strategy_name_list
    assert 'S1R = S1S0 x S0R_0' in strategy_name_list
    assert 'S0R = S0S1 x S1R_0' in strategy_name_list
    assert 'S1R = S1S0 x S0R_1' in strategy_name_list
    assert 'S1R = S1S0 x S0R_2' in strategy_name_list

    # RS = RS x SS
    assert 'RS0 = RS1 x S1S0' in strategy_name_list
    assert 'RS1 = RS0 x S0S1' in strategy_name_list

    # S01R = S01R x RR
    assert 'S01R = S01R x RR_0' in strategy_name_list
    assert 'S01R = S01R x RR_1' in strategy_name_list
    assert 'S01R = S01R x RR_2' in strategy_name_list

    # RR = RS01 x S01R
    assert 'RR = RS01 x S01R' in strategy_name_list

    # RS01 = RR x RS01
    assert 'RS01 = RR x RS01' in strategy_name_list

    if shard_option == ShardOption.SHARD:
        # RR = RS x SR
        assert 'RR = RS0 x S0R' in strategy_name_list
        assert 'RR = RS1 x S1R' in strategy_name_list

        # RS= RR x RS
        assert 'RS0 = RR x RS0' in strategy_name_list
        assert 'RS1 = RR x RS1' in strategy_name_list

    if shard_option == ShardOption.STANDARD:
        # RR = RS x SR
        assert 'RR = RS0 x S0R' in strategy_name_list
        assert 'RR = RS1 x S1R' in strategy_name_list

        # RS= RR x RS
        assert 'RS0 = RR x RS0' in strategy_name_list
        assert 'RS1 = RR x RS1' in strategy_name_list

        # RR = RR x RR
        assert 'RR = RR x RR' in strategy_name_list


@run_on_environment_flag(name='AUTO_PARALLEL')
def test_shard_option():
    for shard_option in [ShardOption.STANDARD, ShardOption.SHARD, ShardOption.FULL_SHARD]:
        check_shard_option(shard_option)


if __name__ == '__main__':
    test_shard_option()
