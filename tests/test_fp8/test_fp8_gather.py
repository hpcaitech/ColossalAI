import torch
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing import assert_close

from colossalai import launch
from colossalai.accelerator import get_accelerator
from colossalai.shardformer.layer._operation import _gather
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
def check_4gpu(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device=get_accelerator().get_current_device())
    output_origin = _gather(x, 0, _get_default_group(), False)
    output_fp8 = _gather(x, 0, _get_default_group(), True)
    assert_close(output_origin, output_fp8, rtol=0.1, atol=0.1)


def run_dist(rank, world_size, port):
    launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_4gpu()


@rerun_if_address_is_in_use()
def test_all_gather():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_all_gather()
