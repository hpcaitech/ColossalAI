import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from colossalai.shardformer.layer._operation import _ring_as_gather
from colossalai.testing import rerun_if_address_is_in_use, spawn


def check_ring_as_gather(rank, world_size, port):
    # gloo on CPU: the ring logic is independent of the device.
    dist.init_process_group("gloo", init_method=f"tcp://localhost:{port}", rank=rank, world_size=world_size)
    torch.manual_seed(0)
    full = torch.randn(2, 4 * world_size, 8)
    weight = torch.randn(6, 8)
    local = full.chunk(world_size, dim=1)[rank].contiguous()

    output, gathered = _ring_as_gather(F.linear, {"input": local}, {"weight": weight}, process_group=dist.group.WORLD)

    assert torch.equal(output, F.linear(full, weight))
    # The backward of the ring linear computes the weight gradient from the gathered input, so the chunk
    # used in every round must be recorded. 2, 3 and 4 ranks exercise the last round, the per-round
    # record and the reuse of the send / recv buffers (a later irecv overwrites a recorded reference).
    assert torch.equal(gathered["input"], full)
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3, 4])
@rerun_if_address_is_in_use()
def test_ring_as_gather(world_size):
    spawn(check_ring_as_gather, world_size)


if __name__ == "__main__":
    test_ring_as_gather(4)
