import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing import assert_close

from colossalai import launch
from colossalai.accelerator import get_accelerator
from colossalai.quantization.fp8 import all_to_all_single_fp8
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


@parameterize("shape", [(4,), (1, 8, 16), (4, 8, 16)])
@parameterize("dtype", [torch.bfloat16, torch.float16])
@parameterize("async_op", [True, False])
def check_all2all(shape, dtype, async_op):
    x = torch.rand(shape, dtype=dtype, device=get_accelerator().get_current_device())
    output = torch.empty_like(x)
    output_fp8 = torch.empty_like(x)
    origin_hanle = dist.all_to_all_single(output, x, group=_get_default_group(), async_op=async_op)
    fp8_handle = all_to_all_single_fp8(output_fp8, x, group=_get_default_group(), async_op=async_op)
    if async_op:
        origin_hanle.wait()
        fp8_handle.wait()
    assert_close(output, output_fp8, rtol=0.1, atol=0.1)


@parameterize("shape", [(8, 8, 16)])
@parameterize("dtype", [torch.bfloat16, torch.float16])
@parameterize("async_op", [True, False])
def check_all2all_uneven(shape, dtype, async_op):
    x = torch.rand(shape, dtype=dtype, device=get_accelerator().get_current_device())
    input_split_sizes = [3, 3, 1, 1]
    if dist.get_rank() in [0, 1]:
        output_split_sizes = [3, 3, 3, 3]
    else:
        output_split_sizes = [1, 1, 1, 1]
    output_shape = list(shape)
    output_shape[0] = sum(output_split_sizes)
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    output_fp8 = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    origin_hanle = dist.all_to_all_single(
        output,
        x,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=_get_default_group(),
        async_op=async_op,
    )
    fp8_handle = all_to_all_single_fp8(
        output_fp8,
        x,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=_get_default_group(),
        async_op=async_op,
    )
    if async_op:
        origin_hanle.wait()
        fp8_handle.wait()
    assert_close(output, output_fp8, rtol=0.1, atol=0.1)


def run_dist(rank, world_size, port):
    launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_all2all()
    check_all2all_uneven()


@rerun_if_address_is_in_use()
def test_all_to_all_single():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_all_to_all_single()
