import torch
import torch.distributed as dist

from colossalai.shardformer.layer._operation import split_forward_gather_backward
from colossalai.testing import rerun_if_address_is_in_use, spawn


def _check_uneven_sequence_split(rank, world_size, port):
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"tcp://127.0.0.1:{port}")
    try:
        # Five tokens cannot be evenly split over two sequence-parallel ranks.
        input_ = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        input_.requires_grad_()

        local = split_forward_gather_backward(input_, dim=1, process_group=dist.group.WORLD)
        assert local.shape == (2, 3, 3)

        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        gathered = torch.cat(gathered, dim=1)
        expected = torch.cat((input_.detach(), torch.zeros(2, 1, 3)), dim=1)
        torch.testing.assert_close(gathered, expected)

        # A downstream operation may touch padded positions.  Those positions
        # must not change the shape or values of the input gradient.  Rank 1's
        # final element is padding, while rank 0's corresponding element is a
        # real token, so use different weights to distinguish the two.
        weight = torch.ones_like(local)
        weight[:, -1] = 10 + rank
        (local * weight).sum().backward()
        expected_grad = torch.ones_like(input_)
        expected_grad[:, 2] = 10
        torch.testing.assert_close(input_.grad, expected_grad)
    finally:
        dist.destroy_process_group()


@rerun_if_address_is_in_use()
def test_uneven_sequence_split_cpu():
    spawn(_check_uneven_sequence_split, nprocs=2)
