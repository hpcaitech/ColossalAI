import pytest
import torch

from colossalai.auto_parallel.meta_profiler import meta_register
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType
from colossalai.testing.utils import clear_cache_before_run, parameterize
from tests.test_auto_parallel.test_tensor_shard.test_metainfo.utils import print_results


@pytest.mark.skipif(torch.__version__ < "1.12.0", reason="need pytorch 1.12.0 or higher for aten level operations")
@clear_cache_before_run()
@parameterize(
    "func",
    [
        torch.nn.functional.softmax,
        torch.nn.functional.relu,
        torch.tanh,
        torch.nn.functional.dropout,
    ],
)
def test_activation_meta_info(func):
    meta_func = meta_register.get(func)
    # construct meta tensors
    input_tensor = torch.rand(256, 1024, device="meta")
    output_tensor = torch.rand(256, 1024, device="meta")
    softmax_dim = 0

    # construct operation data
    input_data = OperationData(name="input", type=OperationDataType.ARG, data=input_tensor)
    output_data = OperationData(name="output", type=OperationDataType.OUTPUT, data=output_tensor)
    softmax_dim_data = OperationData(name="softmax_dim", type=OperationDataType.ARG, data=softmax_dim)

    # construct args and kwargs
    args = [input_data, softmax_dim_data, output_data]
    kwargs = {"inplace": False}

    # estimated results
    compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out = meta_func(*args, **kwargs)

    # actual results
    input_real_tensor = torch.rand(256, 1024, device="cuda")

    input_real_tensor.requires_grad = True

    # fwd
    torch.cuda.reset_peak_memory_stats()
    mem_stamp0 = torch.cuda.memory_allocated()
    output_real_tensor = func(input_real_tensor)
    fwd_allocated = torch.cuda.memory_allocated() - mem_stamp0
    fwd_peak = torch.cuda.max_memory_allocated() - mem_stamp0

    # bwd
    upstream_grad = torch.rand_like(output_real_tensor)
    torch.cuda.reset_peak_memory_stats()
    mem_stamp0 = torch.cuda.memory_allocated()
    torch.autograd.backward(output_real_tensor, upstream_grad)
    bwd_allocated = torch.cuda.memory_allocated() - mem_stamp0
    bwd_peak = torch.cuda.max_memory_allocated() - mem_stamp0

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
    test_activation_meta_info()
