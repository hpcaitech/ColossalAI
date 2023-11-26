import pytest
import torch
import torch.nn as nn

from colossalai._analyzer.fx.graph_module import ColoGraphModule
from colossalai._analyzer.fx.passes.shape_prop import shape_prop_pass
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.tensor_shard.node_handler.matmul_handler import (
    MatMulHandler,
    MatMulType,
    _get_bmm_logical_shape,
    get_matmul_type,
)
from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    OperationData,
    OperationDataType,
    ShardingStrategy,
    StrategiesVector,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.testing.utils import clear_cache_before_run, parameterize


class MatMulModule(nn.Module):
    def forward(self, x1, x2):
        return torch.matmul(x1, x2)


@pytest.mark.skipif(torch.__version__ < "1.12.0", reason="need pytorch 1.12.0 or higher for aten level operations")
@clear_cache_before_run()
@parameterize(
    "tensor_shapes",
    [
        [[8], [8]],  # dot product
        [[4, 8], [8]],  # mat-vec product
        [[4, 8], [8, 16]],  # mat-mat product
        [[8], [8, 16]],  # mat-mat product
        [[8], [4, 8, 16]],  # batched mat-mat product with padding + broadcasting
        [[4, 8, 16], [16]],  # batched mat-mat product with padding + broadcasting
        [[4, 8, 16], [16, 32]],  # batched mat-mat product with broadcasting
        [[4, 8, 16], [1, 16, 32]],  # batched mat-mat product with broadcasting
        [[8, 16], [2, 4, 16, 32]],  # batched mat-mat product with broadcasting
        [[4, 8, 16], [2, 4, 16, 32]],  # batched mat-mat product with broadcasting
        [[1, 8, 16], [2, 4, 16, 32]],  # batched mat-mat product with broadcasting
        [[1, 4, 8, 16], [2, 4, 16, 32]],  # batched mat-mat product with broadcasting
        [[2, 1, 8, 16], [2, 4, 16, 32]],  # batched mat-mat product with broadcasting
        [[2, 4, 8, 16], [2, 4, 16, 32]],  # batched mat-mat product without broadcasting
    ],
)
def test_matmul_node_handler(tensor_shapes):
    input_shape, other_shape = tensor_shapes

    # get output shape
    x1 = torch.rand(*input_shape)
    x2 = torch.rand(*other_shape)
    output_shape = list(torch.matmul(x1, x2).shape)

    # get matmul type
    matmul_type = get_matmul_type(x1.dim(), x2.dim())

    model = MatMulModule()

    tracer = ColoTracer(bias_addition_split=True)
    meta_args = {"x1": x1.to("meta"), "x2": x2.to("meta")}
    graph = tracer.trace(model, meta_args=meta_args)
    gm = ColoGraphModule(model, graph)
    shape_prop_pass(gm, *meta_args.values())
    physical_mesh_id = torch.arange(0, 4)

    print(graph)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    mod_node = list(graph.nodes)[2]
    strategies_vector = StrategiesVector(mod_node)

    # build handler
    handler = MatMulHandler(node=mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    logical_input_shape = input_shape
    logical_other_shape = other_shape
    logical_output_shape = output_shape
    if matmul_type == MatMulType.MM and len(input_shape) == 1:
        logical_input_shape = [1] + input_shape
    elif matmul_type == MatMulType.BMM:
        logical_input_shape, logical_other_shape, logical_output_shape = _get_bmm_logical_shape(
            input_shape, other_shape, handler.transforms
        )
    else:
        logical_input_shape = input_shape

    # check input operation data
    assert mapping["input"].name == "x1"
    assert mapping["input"].data.is_meta
    assert mapping["input"].data.shape == torch.Size(input_shape)
    assert mapping["input"].type == OperationDataType.ARG
    assert mapping["input"].logical_shape == torch.Size(logical_input_shape)

    # check other operation data
    assert mapping["other"].name == "x2"
    assert mapping["other"].data.is_meta
    assert mapping["other"].data.shape == torch.Size(other_shape)
    assert mapping["other"].type == OperationDataType.ARG
    assert mapping["other"].logical_shape == torch.Size(logical_other_shape)

    # check output
    assert mapping["output"].name == "matmul"
    assert mapping["output"].data.is_meta
    assert mapping["output"].data.shape == torch.Size(output_shape)
    assert mapping["output"].type == OperationDataType.OUTPUT
    assert mapping["output"].logical_shape == torch.Size(logical_output_shape)

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # ensure there is no duplicate strategy
    if matmul_type != MatMulType.BMM:
        assert len(set(strategy_name_list)) == len(strategy_name_list), strategy_name_list

    for strategy in strategies_vector:
        strategy: ShardingStrategy
        input_sharding_spec = strategy.get_sharding_spec_by_name("x1")
        other_sharding_spec = strategy.get_sharding_spec_by_name("x2")
        output_sharding_spec = strategy.get_sharding_spec_by_name("matmul")
        if matmul_type == MatMulType.DOT:
            # dot product will produce a scaler
            # results should fulfill:
            # 1. the input and other operands have the same sharding spec
            # 2. the output has no sharding
            assert input_sharding_spec.sharding_sequence == other_sharding_spec.sharding_sequence
            assert len(output_sharding_spec.sharding_sequence) == 0
        elif matmul_type == MatMulType.MV:
            # matrix-vector product should fulfill
            # 1. the last dim of the input and other operands should have the same sharding
            # 2. the first dim of the input and other should have the same sharding
            # 3. the output should have only 1 dim
            assert input_sharding_spec.sharding_sequence[-1] == other_sharding_spec.sharding_sequence[-1]
            assert input_sharding_spec.sharding_sequence[0] == output_sharding_spec.sharding_sequence[0]
            assert len(output_sharding_spec.sharding_sequence) == 1
        elif matmul_type == MatMulType.MM:
            # matrix-matrix multiplication should fulfil
            # 1. if input is a 2D tensor, the 1st dim of input and output should have the same sharding
            # 2. the input's last dim and the first dim of the other should have the same sharding
            # 3. the last dim of the output and other should have the same sharding
            # 4. the input and output should have the same number of dims
            if len(input_shape) == 2:
                assert input_sharding_spec.sharding_sequence[0] == output_sharding_spec.sharding_sequence[0]
            assert input_sharding_spec.sharding_sequence[-1] == other_sharding_spec.sharding_sequence[0]
            assert output_sharding_spec.sharding_sequence[-1] == other_sharding_spec.sharding_sequence[-1]
            assert len(input_sharding_spec.sharding_sequence) == len(output_sharding_spec.sharding_sequence)
        elif matmul_type == MatMulType.BMM:
            # bmm should fulfil
            # 1. of the other tensor is not a 1d tensor, the last dim of other and output have the same sharding
            # 2. if the input has more than 2 dim, the second last dim of input and output have the same sharding
            # 3. if the other have more than 2 dim, the second last dim of other and the last dim of input should have the same sharding
            if len(other_shape) > 1:
                assert other_sharding_spec.sharding_sequence[-1] == output_sharding_spec.sharding_sequence[-1]
            if len(input_shape) > 1:
                if len(other_shape) == 1:
                    assert input_sharding_spec.sharding_sequence[-2] == output_sharding_spec.sharding_sequence[-1]
                else:
                    assert input_sharding_spec.sharding_sequence[-2] == output_sharding_spec.sharding_sequence[-2]
            if len(other_shape) > 2:
                assert other_sharding_spec.sharding_sequence[-2] == input_sharding_spec.sharding_sequence[-1]


if __name__ == "__main__":
    test_matmul_node_handler()
