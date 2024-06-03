import pytest
import torch
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, TrainCycleItem
from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.testing.utils import parameterize, rerun_if_address_is_in_use, spawn
from tests.test_auto_parallel.test_tensor_shard.test_metainfo.utils import mem_test_for_node_strategy, print_results

if torch.__version__ >= "1.12.0":
    from colossalai.auto_parallel.meta_profiler import meta_register


def _batchnorm_module_mem_test(rank, world_size, port):
    """This function is for batchnorm memory test
    Test and print real memory cost and estimated, this test will not be executed except with the tag AUTO_PARALLEL

    Args:
        rank: device rank
        bias: indicate whether conv module need bias
        world_size: number of devices
        port: port for initializing process group
    """
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    model = nn.Sequential(nn.BatchNorm2d(128)).cuda()
    input = torch.rand(4, 128, 64, 64).cuda()
    input.requires_grad = True
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    # index of target node in computation graph
    node_index = 1
    # total number of target node strategies
    strategy_number = 9
    mem_test_for_node_strategy(
        rank=rank,
        model=model,
        device_mesh=device_mesh,
        node_index=node_index,
        strategy_number=strategy_number,
        input_args=[input],
        meta_arg_names=["input"],
    )


@run_on_environment_flag(name="AUTO_PARALLEL")
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_batchnorm_meta_concrete_info_match():
    spawn(_batchnorm_module_mem_test, 4)


@pytest.mark.skipif(torch.__version__ < "1.12.0", reason="need pytorch 1.12.0 or higher for aten level operations")
@parameterize(
    "tensor_shape",
    [
        [256, 1024],
        [1024, 256],
    ],
)
def test_layernorm_meta_info(tensor_shape):
    meta_func = meta_register.get(torch.nn.LayerNorm)

    # construct input
    input_tensor = torch.rand(*tensor_shape, device="meta")
    output_tensor = torch.rand(*tensor_shape, device="meta")
    weight_tensor = torch.rand(tensor_shape[1], device="meta")
    bias_tensor = torch.rand(tensor_shape[1], device="meta")

    # construct operation data
    input_data = OperationData(name="input", type=OperationDataType.ARG, data=input_tensor)

    output_data = OperationData(name="output", type=OperationDataType.OUTPUT, data=output_tensor)

    weight_data = OperationData(name="weight", type=OperationDataType.PARAM, data=weight_tensor)

    bias_data = OperationData(name="bias", type=OperationDataType.PARAM, data=bias_tensor)

    # construct args and kwargs
    args = [input_data, output_data, weight_data, bias_data]
    kwargs = {"inplace": False}

    # estimated results
    compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out = meta_func(*args, **kwargs)

    # actual results
    input_real_tensor = torch.rand(*tensor_shape, device="cuda:0")

    input_real_tensor.requires_grad = True

    ln_module = torch.nn.LayerNorm(tensor_shape[1]).cuda()

    # fwd
    torch.cuda.reset_peak_memory_stats()
    mem_stamp0 = torch.cuda.memory_allocated()
    output_real_tensor = ln_module(input_real_tensor)
    fwd_allocated = torch.cuda.memory_allocated() - mem_stamp0
    fwd_peak = torch.cuda.max_memory_allocated() - mem_stamp0

    # bwd
    upstream_grad = torch.rand_like(output_real_tensor)
    torch.cuda.reset_peak_memory_stats()
    mem_stamp0 = torch.cuda.memory_allocated()
    torch.autograd.backward(output_real_tensor, upstream_grad)
    bwd_allocated = torch.cuda.memory_allocated() - mem_stamp0
    bwd_peak = torch.cuda.max_memory_allocated() - mem_stamp0

    compute_cost: TrainCycleItem
    memory_cost: TrainCycleItem

    print_results(
        [input_real_tensor],
        [output_real_tensor],
        compute_cost,
        memory_cost,
        fwd_allocated,
        fwd_peak,
        bwd_allocated,
        bwd_peak,
    )


if __name__ == "__main__":
    test_batchnorm_meta_concrete_info_match()
    test_layernorm_meta_info()
