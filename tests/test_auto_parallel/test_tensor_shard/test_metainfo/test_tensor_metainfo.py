import pytest
import torch
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType
from colossalai.testing.utils import clear_cache_before_run
from tests.test_auto_parallel.test_tensor_shard.test_metainfo.utils import print_results

if torch.__version__ >= "1.12.0":
    from colossalai.auto_parallel.meta_profiler import meta_register


class SplitModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x.split(512, dim=0)


@pytest.mark.skipif(torch.__version__ < "1.12.0", reason="need pytorch 1.12.0 or higher for aten level operations")
@clear_cache_before_run()
def test_tensor_meta_info():
    """test tensor related meta information
    We will just use torch.Tensor.split for the test
    """
    meta_func = meta_register.get(torch.Tensor.split)

    # construct meta tensors
    input_tensor = torch.rand(1024, 1024, device="meta")
    output_tensor = input_tensor.split(512, dim=0)

    # construct operation data
    input_data = OperationData(
        name="input",
        data=input_tensor,
        type=OperationDataType.ARG,
        logical_shape=input_tensor.shape,
    )
    output_data = OperationData(
        name="output",
        data=output_tensor,
        type=OperationDataType.OUTPUT,
        logical_shape=input_tensor.shape,
    )
    split_info_data = OperationData(
        name="split_info",
        type=OperationDataType.ARG,
        data=0,
        logical_shape=None,
    )

    # construct args
    args = [input_data, output_data, split_info_data]
    kwargs = {"inplace": False}

    # estimated results
    compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out = meta_func(*args, **kwargs)

    # actual results
    model = SplitModule()
    input_real_tensor = torch.rand(1024, 1024).cuda()

    input_real_tensor.requires_grad = True

    # fwd
    torch.cuda.reset_peak_memory_stats()
    mem_stamp0 = torch.cuda.memory_allocated()
    output_real_tensor = model(input_real_tensor)
    fwd_allocated = torch.cuda.memory_allocated() - mem_stamp0
    fwd_peak = torch.cuda.max_memory_allocated() - mem_stamp0

    # bwd
    upstream_grad = [torch.rand_like(tensor) for tensor in output_real_tensor]
    torch.cuda.reset_peak_memory_stats()
    mem_stamp0 = torch.cuda.memory_allocated()
    torch.autograd.backward(output_real_tensor, upstream_grad)
    bwd_allocated = torch.cuda.memory_allocated() - mem_stamp0
    bwd_peak = torch.cuda.max_memory_allocated() - mem_stamp0

    print_results(
        [input_real_tensor],
        output_real_tensor,
        compute_cost,
        memory_cost,
        fwd_allocated,
        fwd_peak,
        bwd_allocated,
        bwd_peak,
    )


if __name__ == "__main__":
    test_tensor_meta_info()
