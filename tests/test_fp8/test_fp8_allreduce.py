import torch
import torch.distributed
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
@parameterize("dtype", [torch.float16])
def check_4gpu(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device=get_accelerator().get_current_device())
    x_fp8 = x.clone()
    torch.distributed.all_reduce(x)
    all_reduce_fp8(x_fp8)
    assert_close(x, x_fp8, rtol=0.1, atol=0.1)


def run_dist(rank, world_size, port):
    launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_4gpu()


@rerun_if_address_is_in_use()
def test_all_reduce():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_all_reduce()
