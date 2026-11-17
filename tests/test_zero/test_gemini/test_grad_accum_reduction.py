"""Gradient accumulation must keep an already averaged gradient unchanged across later reductions."""

import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import spawn


def _run(rank, world_size, port, keep_gathered, asynchronous):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    try:
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.125)
        plugin = GeminiPlugin(
            chunk_config_dict={world_size: dict(chunk_size=2, keep_gathered=keep_gathered)},
            precision="bf16",
            enable_gradient_accumulation=True,
            enable_async_reduce=asynchronous,
        )
        booster = Booster(plugin=plugin)
        optimizer = HybridAdam(model.parameters(), lr=1e-3, betas=(0.0, 0.999), weight_decay=0.0)
        model, optimizer, *_ = booster.boost(model, optimizer)
        # Unequal micro-batches make the weighting of the previous gradient visible. After each backward,
        # Gemini averages across ranks. Two backwards must therefore give [world_size + 3, 2].
        for micro in range(2):
            x = torch.tensor([[rank + 1.0 + 2 * micro, 1.0]], device="cuda")
            booster.backward(model(x).float().sum(), optimizer)
        optimizer.step()
        state = optimizer.collect_states(0, only_rank_0=False)
        expected = torch.tensor([[world_size + 3.0, 2.0]])
        torch.testing.assert_close(state["exp_avg"], expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Gemini requires CUDA")
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("keep_gathered", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_accumulated_gradient_reduction(world_size, keep_gathered, asynchronous):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CUDA devices")
    spawn(_run, world_size, keep_gathered=keep_gathered, asynchronous=asynchronous)
