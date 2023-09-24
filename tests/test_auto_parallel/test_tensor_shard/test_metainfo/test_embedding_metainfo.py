import pytest
import torch

from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType
from colossalai.testing.utils import clear_cache_before_run
from tests.test_auto_parallel.test_tensor_shard.test_metainfo.utils import print_results

if torch.__version__ >= "1.12.0":
    from colossalai.auto_parallel.meta_profiler import meta_register


@pytest.mark.skipif(torch.__version__ < "1.12.0", reason="need pytorch 1.12.0 or higher for aten level operations")
@clear_cache_before_run()
def test_embedding_meta_info():
    meta_func = meta_register.get(torch.nn.Embedding)

    # construct meta tensors
    input_tensor = torch.randint(0, 50256, (8, 1024), device="meta")
    weight_tensor = torch.rand(50257, 1024, device="meta")
    output_tensor = torch.rand(8, 1024, 1024, device="meta")

    # construct operation data
    input_data = OperationData(name="input", type=OperationDataType.ARG, data=input_tensor)

    weight_data = OperationData(name="weight", type=OperationDataType.PARAM, data=weight_tensor)

    output_data = OperationData(name="output", type=OperationDataType.OUTPUT, data=output_tensor)

    # construct args and kwargs
    args = [input_data, weight_data, output_data]
    kwargs = {"inplace": False}

    # estimated results
    compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out = meta_func(*args, **kwargs)

    # actual results
    input_real_tensor = torch.randint(0, 50256, (8, 1024), device="cuda")
    embedding_module = torch.nn.Embedding(50257, 1024).cuda()

    # fwd
    torch.cuda.reset_peak_memory_stats()
    mem_stamp0 = torch.cuda.memory_allocated()
    output_real_tensor = embedding_module(input_real_tensor)
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
    test_embedding_meta_info()
