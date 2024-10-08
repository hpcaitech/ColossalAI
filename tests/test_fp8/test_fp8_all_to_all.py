import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing import assert_close

from colossalai import launch
from colossalai.accelerator import get_accelerator
from colossalai.quantization.fp8 import _all_to_all_fp8
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


@parameterize("shape", [(16, 8, 4)])
@parameterize("scatter_dim", [0, 1, 2])
@parameterize("dtype", [torch.bfloat16, torch.float16])
@parameterize("fp8_format", ["e4m3", "e5m2"])
def check_4gpu(shape, scatter_dim, dtype, fp8_format):
    world_size = dist.get_world_size()
    input_tensor = torch.rand(shape, dtype=dtype, device=get_accelerator().get_current_device())
    input_tensor_list = list(torch.chunk(input_tensor, world_size, scatter_dim))
    input_tensor_list = [x.contiguous() for x in input_tensor_list]
    output_tensor_list_fp8 = [torch.empty_like(x) for x in input_tensor_list]
    output_tensor_list = [torch.empty_like(x) for x in input_tensor_list]
    _all_to_all_fp8(output_tensor_list_fp8, input_tensor_list, group=_get_default_group(), fp8_format=fp8_format)
    dist.all_to_all(output_tensor_list, input_tensor_list, group=_get_default_group())
    assert_close(output_tensor_list_fp8, output_tensor_list, rtol=0.1, atol=0.1)


def run_dist(rank, world_size, port):
    launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_4gpu()


@rerun_if_address_is_in_use()
def test_all_to_all():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_all_to_all()
