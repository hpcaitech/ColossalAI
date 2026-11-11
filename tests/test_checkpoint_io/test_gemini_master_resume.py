import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import rerun_if_address_is_in_use, spawn


class TiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(7, 11)
        self.second = torch.nn.Linear(11, 5)
        self.alias = torch.nn.Linear(7, 11)
        self.alias.weight = self.first.weight
        self.alias.bias = self.first.bias

    def forward(self, inputs):
        return self.second(torch.nn.functional.gelu(self.first(inputs)))


def _run_gemini_resume(rank, world_size, port, accumulation, shard, steps_before, checkpoint_dir):
    torch.set_num_threads(1)
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost", seed=42)
    try:

        def create():
            model = TiedModel()
            optimizer = HybridAdam(model.parameters(), lr=1e-4)
            booster = Booster(
                plugin=GeminiPlugin(
                    chunk_config_dict={world_size: {"chunk_size": 256, "keep_gathered": world_size == 1}},
                    placement_policy="static",
                    precision="bf16",
                    enable_gradient_accumulation=accumulation,
                    enable_all_optimization=False,
                )
            )
            model, optimizer, _, _, _ = booster.boost(model, optimizer)
            return booster, model, optimizer

        booster, model, optimizer = create()
        inputs = torch.randn(3, 7, device="cuda")

        def step(booster, model, optimizer):
            optimizer.zero_grad()
            loss = model(inputs).float().square().mean()
            booster.backward(loss, optimizer)
            optimizer.step()
            return loss.detach().clone()

        def masters(optimizer):
            result = []
            for group in optimizer.param_groups:
                for param in group["params"]:
                    begin, end = optimizer.param_to_range[param]
                    result.append(optimizer.param_to_chunk16[param].paired_chunk.payload[begin:end].clone())
            return result

        for _ in range(steps_before):
            step(booster, model, optimizer)
        master_before = masters(optimizer)
        if steps_before:
            assert any(not torch.equal(p, p.bfloat16().float()) for p in master_before)
        optimizer_snapshot = optimizer.state_dict(only_rank_0=False)
        assert optimizer_snapshot["state"]
        assert all("_colossalai_master_param" in state for state in optimizer_snapshot["state"].values())
        assert all("_colossalai_master_param" not in state for state in optimizer.optim.state.values())
        snapshot = model.state_dict(only_rank_0=False)
        # Preserve Gemini's existing export behavior: gathered reused master chunks stay FP32,
        # while sharded chunks or non-reused working chunks export in the working precision.
        expected_dtype = torch.float32 if not accumulation and world_size == 1 else torch.bfloat16
        assert all(value.dtype == expected_dtype for value in snapshot.values())
        saved_snapshot = {name: value.clone() for name, value in snapshot.items()}
        model_path = f"{checkpoint_dir}/model" + ("" if shard else ".pt")
        optimizer_path = f"{checkpoint_dir}/optimizer" + ("" if shard else ".pt")
        booster.save_model(model, model_path, shard=shard)
        booster.save_optimizer(optimizer, optimizer_path, shard=shard, size_per_shard=0.001)
        dist.barrier()
        expected_losses = [step(booster, model, optimizer) for _ in range(2)]
        for name, value in snapshot.items():
            torch.testing.assert_close(value, saved_snapshot[name], atol=0, rtol=0)
        expected_masters = masters(optimizer)
        restored_booster, restored_model, restored_optimizer = create()
        restored_booster.load_model(restored_model, model_path)
        restored_booster.load_optimizer(restored_optimizer, optimizer_path)
        assert restored_model.unwrap().first.weight is restored_model.unwrap().alias.weight
        assert all("_colossalai_master_param" not in state for state in restored_optimizer.optim.state.values())
        for actual, expected in zip(masters(restored_optimizer), master_before):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for expected in expected_losses:
            torch.testing.assert_close(
                step(restored_booster, restored_model, restored_optimizer), expected, atol=0, rtol=0
            )
        for actual, expected in zip(masters(restored_optimizer), expected_masters):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        # Old optimizer files have no master metadata and must remain loadable.
        legacy = restored_optimizer.state_dict(only_rank_0=False)
        for state in legacy["state"].values():
            state.pop("_colossalai_master_param", None)
        legacy_masters = masters(restored_optimizer)
        restored_optimizer.load_state_dict(legacy)
        for actual, expected in zip(masters(restored_optimizer), legacy_masters):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("accumulation", [False, True])
@pytest.mark.parametrize("shard", [False, True])
@pytest.mark.parametrize("steps_before", [0, 1])
@rerun_if_address_is_in_use()
def test_gemini_master_resume(world_size, accumulation, shard, steps_before, tmp_path):
    spawn(
        _run_gemini_resume,
        world_size,
        accumulation=accumulation,
        shard=shard,
        steps_before=steps_before,
        checkpoint_dir=str(tmp_path),
    )
