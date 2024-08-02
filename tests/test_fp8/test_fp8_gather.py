import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing import assert_close

from colossalai import launch
from colossalai.accelerator import get_accelerator
from colossalai.quantization.fp8 import gather_fp8
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


@parameterize(
    "shape",
    [
        (3, 7),
        (2, 1),
        (1, 2),
        (2, 2),
        (4, 2),
        (5,),
        (4,),
        (2,),
    ],
)
@parameterize("dtype", [torch.bfloat16, torch.float16])
@parameterize("fp8_format", ["e4m3", "e5m2"])
def check_4gpu(shape, dtype, fp8_format):
    world_size = dist.get_world_size()
    x = torch.rand(shape, dtype=dtype, device=get_accelerator().get_current_device())
    output_list = [torch.empty_like(x) for _ in range(world_size)]
    output_list_fp8 = [torch.empty_like(x) for _ in range(world_size)]
    gather_fp8(output_list_fp8, x, group=_get_default_group(), fp8_format=fp8_format)
    dist.all_gather(output_list, x, group=_get_default_group())
    assert_close(output_list, output_list_fp8, rtol=0.1, atol=0.1)


def run_dist(rank, world_size, port):
    launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_4gpu()


@rerun_if_address_is_in_use()
def test_all_gather():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_all_gather()
