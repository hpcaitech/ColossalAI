import pprint

import pytest
import torch
import torch.nn as nn
from typing_extensions import Self

from colossalai.auto_parallel.tensor_shard.node_handler import LinearFunctionHandler, LinearModuleHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    OperationData,
    OperationDataType,
    ShardingStrategy,
    StrategiesVector,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.fx.meta_profiler import MetaInfo, meta_register
from colossalai.testing.utils import parameterize


@pytest.mark.skipif(torch.__version__ < '1.12.0', reason='PyTorch version is too low')
@parameterize('bias', [True, False])
def test_linear_module_handler(bias):
    model = nn.Sequential(nn.Linear(16, 64, bias=bias).to('meta'))

    tracer = ColoTracer()
    graph = tracer.trace(model, meta_args={"input": torch.rand(2, 2, 2, 16).to('meta')})
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    linear_mod_node = list(graph.nodes)[1]
    strategies_vector = StrategiesVector(linear_mod_node)

    # build handler
    handler = LinearModuleHandler(node=linear_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # build strategy
    strategies_vector = handler.register_strategy(compute_resharding_cost=False)

    # assert module is registered
    assert meta_register.has(linear_mod_node.graph.owning_module.get_submodule(linear_mod_node.target).__class__)

    # check metainfo
    for strategy in strategies_vector:
        strategy: ShardingStrategy
        try:
            metainfo = MetaInfo(strategy,
                                linear_mod_node.graph.owning_module.get_submodule(linear_mod_node.target).__class__)

            # this region is for debugging
            print("compute cost:")
            pprint.pprint(metainfo.compute_cost)
            print("memory cost:")
            pprint.pprint(metainfo.memory_cost)
            print(f"device mesh: {device_mesh.mesh_shape}")
            for opdata, spec in strategy.sharding_specs.items():
                print(opdata.type, opdata.data.shape, spec.sharding_sequence)
        except:
            raise RuntimeError(f"Failed to compute metainfo for {strategy}")


if __name__ == '__main__':
    test_linear_module_handler(bias=True)
    test_linear_module_handler(bias=False)
