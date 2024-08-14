import torch
import torch.distributed as dist
from torch.testing import assert_close

from colossalai import launch
from colossalai.accelerator import get_accelerator
from colossalai.quantization.fp8 import all_reduce_fp8
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


@parameterize(
    "shape",
    [
        (3, 7),
        (4, 7),
        (7, 4),
        (8, 9),
        (3),
        (7,),
        (8,),
    ],
)
@parameterize("dtype", [torch.float16, torch.bfloat16])
@parameterize("fp8_format", ["e4m3", "e5m2"])
@parameterize("async_op", [True, False])
def check_4gpu(shape, dtype, fp8_format, async_op):
    x = torch.rand(shape, dtype=dtype, device=get_accelerator().get_current_device())
    x_fp8 = x.clone()
    origin_handle = dist.all_reduce(x, async_op=async_op)
    fp8_handle = all_reduce_fp8(x_fp8, fp8_format=fp8_format, async_op=async_op)
    if async_op:
        origin_handle.wait()
        fp8_handle.wait()
    assert_close(x, x_fp8, rtol=0.1, atol=0.1)

    origin_handle = dist.all_reduce(x, op=dist.ReduceOp.AVG, async_op=async_op)
    fp8_handle = all_reduce_fp8(x_fp8, op=dist.ReduceOp.AVG, fp8_format=fp8_format, async_op=async_op)
    if async_op:
        origin_handle.wait()
        fp8_handle.wait()
    assert_close(x, x_fp8, rtol=0.1, atol=0.1)


def run_dist(rank, world_size, port):
    launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_4gpu()


@rerun_if_address_is_in_use()
def test_all_reduce():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_all_reduce()
