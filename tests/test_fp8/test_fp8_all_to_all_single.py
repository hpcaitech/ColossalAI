import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing import assert_close

from colossalai import launch
from colossalai.accelerator import get_accelerator
from colossalai.quantization.fp8 import all_to_all_single_fp8
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn

dist.all_to_all_single


@parameterize("shape", [(4), (8, 7), (4, 8, 16)])
@parameterize("dtype", [torch.bfloat16, torch.float16])
@parameterize("fp8_format", ["e4m3", "e5m2"])
def check_4gpu(shape, dtype, fp8_format):
    x = torch.rand(shape, dtype=dtype, device=get_accelerator().get_current_device())
    output = torch.empty_like(x)
    output_fp8 = torch.empty_like(x)
    all_to_all_single_fp8(output_fp8, x, group=_get_default_group(), fp8_format=fp8_format)
    dist.all_to_all_single(output, x, group=_get_default_group())
    assert_close(output, output_fp8, rtol=0.1, atol=0.1)


def run_dist(rank, world_size, port):
    launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_4gpu()


@rerun_if_address_is_in_use()
def test_all_to_all_single():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_all_to_all_single()
