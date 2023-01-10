import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.testing import assert_close


def assert_equal(a: Tensor, b: Tensor):
    assert torch.all(a == b), f'expected a and b to be equal but they are not, {a} vs {b}'


def assert_not_equal(a: Tensor, b: Tensor):
    assert not torch.all(a == b), f'expected a and b to be not equal but they are, {a} vs {b}'


def assert_close_loose(a: Tensor, b: Tensor, rtol: float = 1e-3, atol: float = 1e-3):
    assert_close(a, b, rtol=rtol, atol=atol)


def assert_equal_in_group(tensor: Tensor, process_group: ProcessGroup = None):
    # all gather tensors from different ranks
    world_size = dist.get_world_size(process_group)
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=process_group)

    # check if they are equal one by one
    for i in range(world_size - 1):
        a = tensor_list[i]
        b = tensor_list[i + 1]
        assert torch.all(a == b), f'expected tensors on rank {i} and {i + 1} to be equal but they are not, {a} vs {b}'
