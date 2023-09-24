import pytest
import torch

from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, TrainCycleItem
from colossalai.testing.utils import clear_cache_before_run
from tests.test_auto_parallel.test_tensor_shard.test_metainfo.utils import print_results

if torch.__version__ >= "1.12.0":
    from colossalai.auto_parallel.meta_profiler import meta_register


@pytest.mark.skipif(torch.__version__ < "1.12.0", reason="need pytorch 1.12.0 or higher for aten level operations")
@clear_cache_before_run()
def test_where_meta_info():
    meta_func = meta_register.get(torch.where)

    # construct meta tensors
    condition_tensor = torch.rand(1, 1, 1024, 1024) > 0.5
    condition_tensor = condition_tensor.to(device="meta")
    x_tensor = torch.rand(8, 16, 1024, 1024, device="meta")
    y_tensor = torch.tensor(0, device="meta")
    output_tensor = torch.rand(8, 16, 1024, 1024)

    # construct operation data
    condition_data = OperationData(
        name="condition",
        data=condition_tensor,
        type=OperationDataType.ARG,
        logical_shape=condition_tensor.shape,
    )
    x_data = OperationData(
        name="x",
        data=x_tensor,
        type=OperationDataType.ARG,
        logical_shape=x_tensor.shape,
    )
    y_data = OperationData(
        name="y",
        data=y_tensor,
        type=OperationDataType.ARG,
        logical_shape=y_tensor.shape,
    )
    output_data = OperationData(
        name="output",
        data=output_tensor,
        type=OperationDataType.OUTPUT,
        logical_shape=output_tensor.shape,
    )

    # construct args and kwargs
    args = [condition_data, x_data, y_data, output_data]
    kwargs = {"inplace": False}

    # estimated results
    compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out = meta_func(*args, **kwargs)

    # actual results
    condition_real_tensor = torch.rand(1, 1, 1024, 1024) > 0.5
    condition_real_tensor = condition_real_tensor.to(device="cuda")
    x_real_tensor = torch.rand(8, 16, 1024, 1024, device="cuda")
    y_real_tensor = torch.tensor(0.0, device="cuda")

    x_real_tensor.requires_grad = True
    y_real_tensor.requires_grad = True

    # fwd
    torch.cuda.reset_peak_memory_stats()
    mem_stamp0 = torch.cuda.memory_allocated()
    output_real_tensor = torch.where(condition_real_tensor, x_real_tensor, y_real_tensor)
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
        [condition_real_tensor, x_real_tensor, y_real_tensor],
        [output_real_tensor],
        compute_cost,
        memory_cost,
        fwd_allocated,
        fwd_peak,
        bwd_allocated,
        bwd_peak,
    )


if __name__ == "__main__":
    test_where_meta_info()
