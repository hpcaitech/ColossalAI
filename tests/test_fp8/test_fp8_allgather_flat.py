import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing import assert_close

from colossalai import launch
from colossalai.accelerator import get_accelerator
from colossalai.quantization.fp8 import all_gather_into_tensor_flat_fp8
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


@parameterize("shape", [(3, 7), (2, 1), (1, 2), (2, 2), (4, 2), (5,), (4,), (2,)])
@parameterize("dtype", [torch.bfloat16, torch.float16])
def check_4gpu(shape, dtype):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    x = torch.rand(shape, dtype=dtype, device=get_accelerator().get_current_device())
    flat_padded_x = x.view(-1)
    if flat_padded_x.size(0) % world_size != 0:
        pad_size = world_size - flat_padded_x.size(0) % world_size
        flat_padded_x = F.pad(flat_padded_x, (0, pad_size))
    output = torch.empty_like(flat_padded_x)
    chunk = flat_padded_x.chunk(world_size)[rank].clone()
    all_gather_into_tensor_flat_fp8(output, chunk, x.shape, group=_get_default_group())
    assert_close(output[: x.numel()], x.view(-1), rtol=0.1, atol=0.1)


def run_dist(rank, world_size, port):
    launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_4gpu()


@rerun_if_address_is_in_use()
def test_all_gather_flat():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_all_gather_flat()
