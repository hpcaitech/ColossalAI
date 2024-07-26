import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing import assert_close

from colossalai import launch
from colossalai.accelerator import get_accelerator
from colossalai.quantization.fp8 import all_reduce_fp8
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


@parameterize("shape", [(3, 7), (2, 1), (1, 2), (2, 2), (4, 2), (5,), (4,), (2,)])
@parameterize("dtype", [torch.bfloat16, torch.float16])
def check_4gpu(shape, dtype):
    dist.get_world_size()
    dist.get_rank()
    x = torch.rand(shape, dtype=dtype, device=get_accelerator().get_current_device())
    flat_padded_x = x.view(-1)

    ground_truth = flat_padded_x.clone()
    all_reduce_fp8(flat_padded_x, group=_get_default_group())

    assert_close(ground_truth, flat_padded_x, rtol=0.1, atol=0.1)


def run_dist(rank, world_size, port):
    launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_4gpu()


@rerun_if_address_is_in_use()
def test_all_reduce():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_all_reduce()
