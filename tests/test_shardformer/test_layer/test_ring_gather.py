"""CPU regression for the input snapshots retained by ring sequence parallelism."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from colossalai.shardformer.layer._operation import _ring_as_gather


def _check_ring_gather(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        # Distinct shards expose incorrect rank ordering and reused receive buffers.
        chunks = [torch.arange(12, dtype=torch.float64).reshape(1, 3, 4) + 100 * r for r in range(world_size)]
        local = chunks[rank]
        # Dyadic values make this a bit-exact communication test, independent
        # of the GEMM implementation's floating-point reduction order.
        weight = torch.arange(20, dtype=torch.float64).reshape(5, 4) / 8
        output, saved = _ring_as_gather(F.linear, {"input": local}, {"weight": weight}, dist.group.WORLD)
        expected_input = torch.cat(chunks, dim=1)
        torch.testing.assert_close(output, F.linear(expected_input, weight), rtol=0, atol=0)
        torch.testing.assert_close(saved["input"], expected_input, rtol=0, atol=0)
        # Backward's weight gradient must use the same inputs as the forward.
        grad = torch.arange(output.numel(), dtype=torch.float64).reshape_as(output)
        actual_grad = grad.flatten(0, 1).T @ saved["input"].flatten(0, 1)
        reference_weight = weight.clone().requires_grad_()
        F.linear(expected_input, reference_weight).backward(grad)
        torch.testing.assert_close(actual_grad, reference_weight.grad, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_ring_gather_saved_inputs(world_size, tmp_path):
    mp.spawn(_check_ring_gather, args=(world_size, (tmp_path / "rendezvous").as_uri()), nprocs=world_size)
