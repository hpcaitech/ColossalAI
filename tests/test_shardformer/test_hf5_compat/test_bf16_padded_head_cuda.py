"""Two-rank physical vocabulary-head acceptance; not full-model coverage."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from colossalai.shardformer.layer import VocabParallelLMHead1D
from tests.test_shardformer.test_hf5_compat.bf16_reference import grouped_column


def _worker(rank, rendezvous, seed):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(seed)
        reference = torch.nn.Linear(256, 51866, bias=False, device="cuda", dtype=torch.bfloat16)
        candidate = VocabParallelLMHead1D.from_native_module(
            torch.nn.Linear(256, 51866, bias=False, device="cuda", dtype=torch.bfloat16),
            process_group=dist.group.WORLD,
            gather_output=True,
            make_vocab_size_divisible_by=64,
        )
        dist.broadcast(reference.weight.data, src=0)
        padded = torch.nn.functional.pad(reference.weight.detach(), (0, 0, 0, 102))
        with torch.no_grad():
            candidate.weight.copy_(padded.chunk(2, 0)[rank])
        assert candidate.weight.shape == (25984, 256)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01)
        for _ in range(2):
            ref_optimizer.zero_grad()
            optimizer.zero_grad()
            x = torch.randn(1, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            dist.broadcast(x.data, src=0)
            actual_x = x.detach().clone().requires_grad_(True)
            expected = grouped_column(x, reference.weight, padding_multiple=64)
            actual = candidate(actual_x)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            labels = torch.tensor([0, 1], device="cuda")
            ref_loss = torch.nn.functional.cross_entropy(expected.reshape(-1, 51866).float(), labels)
            loss = torch.nn.functional.cross_entropy(actual.reshape(-1, 51866).float(), labels)
            torch.testing.assert_close(loss, ref_loss, atol=0, rtol=0)
            ref_loss.backward()
            loss.backward()
            torch.testing.assert_close(actual_x.grad, x.grad, atol=0.005, rtol=0.005)
            parts = [torch.empty_like(candidate.weight.grad) for _ in range(2)]
            dist.all_gather(parts, candidate.weight.grad)
            full_grad = torch.cat(parts, 0)
            torch.testing.assert_close(full_grad[:51866], reference.weight.grad, atol=0.005, rtol=0.005)
            assert torch.count_nonzero(full_grad[51866:]) == 0
            ref_optimizer.step()
            optimizer.step()
            parts = [torch.empty_like(candidate.weight) for _ in range(2)]
            dist.all_gather(parts, candidate.weight.detach())
            full_weight = torch.cat(parts, 0)
            torch.testing.assert_close(full_weight[:51866], reference.weight, atol=0.005, rtol=0.005)
            assert torch.count_nonzero(full_weight[51866:]) == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two real CUDA GPUs and NCCL")
@pytest.mark.parametrize("seed", [42, 123])
def test_padded_vocabulary_head_cuda(tmp_path, seed):
    mp.spawn(_worker, args=((tmp_path / "nccl-init").as_uri(), seed), nprocs=2, join=True)
