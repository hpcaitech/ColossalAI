import torch
from torch.distributed import reduce_scatter
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing import assert_close

from colossalai import launch
from colossalai.accelerator import get_accelerator
from colossalai.quantization.fp8 import reduce_scatter_fp8
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


@parameterize("shape", [(16, 8, 4)])
@parameterize("scatter_dim", [0, 1, 2])
@parameterize("dtype", [torch.bfloat16, torch.float16])
@parameterize("fp8_format", ["e4m3", "e5m2"])
@parameterize("async_op", [True, False])
def check_4gpu(shape, scatter_dim, dtype, fp8_format, async_op):
    x = torch.rand(shape, dtype=dtype, device=get_accelerator().get_current_device())
    input_list = list(torch.chunk(x, dim=scatter_dim, chunks=4))
    input_list = [t.contiguous() for t in input_list]
    output_origin = torch.empty_like(input_list[0])
    output_fp8 = torch.empty_like(input_list[0])
    origin_handle = reduce_scatter(output_origin, input_list, group=_get_default_group(), async_op=async_op)
    fp8_handle = reduce_scatter_fp8(
        output_fp8, input_list, group=_get_default_group(), fp8_format=fp8_format, async_op=async_op
    )
    if async_op:
        origin_handle.wait()
        fp8_handle.wait()
    assert_close(output_origin, output_fp8, rtol=0.1, atol=0.1)


def run_dist(rank, world_size, port):
    launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_4gpu()


@rerun_if_address_is_in_use()
def test_reduce_scatter():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_reduce_scatter()
