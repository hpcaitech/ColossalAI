import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing import assert_close

from colossalai import launch
from colossalai.accelerator import get_accelerator
from colossalai.shardformer.layer._operation import _all_to_all_single
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


@parameterize("shape", [(1, 8, 16)])
@parameterize("scatter_dim", [1, 2])
@parameterize("dtype", [torch.bfloat16, torch.float16])
def check_4gpu(shape, scatter_dim, dtype):
    world_size = dist.get_world_size()
    x = torch.rand(shape, dtype=dtype, device=get_accelerator().get_current_device())
    output_origin = _all_to_all_single(x, world_size, _get_default_group(), scatter_dim, 0, False)
    output_fp8 = _all_to_all_single(x, world_size, _get_default_group(), scatter_dim, 0, True)
    assert_close(output_origin, output_fp8, rtol=0.1, atol=0.1)


def run_dist(rank, world_size, port):
    launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_4gpu()


@rerun_if_address_is_in_use()
def test_all_to_all_single():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_all_to_all_single()
