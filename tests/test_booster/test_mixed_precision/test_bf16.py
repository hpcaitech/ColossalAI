from copy import deepcopy

import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.booster import Booster
from colossalai.booster.mixed_precision import BF16MixedPrecision
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.interface import OptimizerWrapper
from colossalai.testing import spawn


def _run_bf16(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost", seed=42)
    try:
        reference = torch.nn.Linear(7, 5).cuda()
        candidate = deepcopy(reference)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
        booster = Booster(plugin=TorchDDPPlugin(), mixed_precision="bf16")
        candidate, optimizer, criterion, _, _ = booster.boost(candidate, optimizer, torch.nn.MSELoss())
        assert isinstance(optimizer, OptimizerWrapper)
        assert not isinstance(optimizer.optim, OptimizerWrapper)
        assert all(p.dtype == torch.float32 for p in candidate.parameters())
        inputs = torch.randn(3, 7, device="cuda")
        targets = torch.randn(3, 5, device="cuda")
        for _ in range(3):
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                expected = reference(inputs)
                reference_loss = torch.nn.functional.mse_loss(expected, targets)
            actual = candidate(inputs)
            assert actual.dtype == torch.bfloat16
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            loss = criterion(actual, targets)
            torch.testing.assert_close(loss, reference_loss, atol=0, rtol=0)
            booster.backward(loss, optimizer)
            reference_loss.backward()
            for param, ref in zip(candidate.unwrap().parameters(), reference.parameters()):
                torch.testing.assert_close(param.grad, ref.grad, atol=0, rtol=0)
            optimizer.step()
            reference_optimizer.step()
            for param, ref in zip(candidate.unwrap().parameters(), reference.parameters()):
                torch.testing.assert_close(param, ref, atol=0, rtol=0)
        wrapped, optional_optimizer, optional_criterion = BF16MixedPrecision().configure(reference)
        assert optional_optimizer is None and optional_criterion is None
        assert wrapped.unwrap() is reference
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
def test_bf16_ddp_matches_native_autocast(world_size):
    spawn(_run_bf16, world_size)
