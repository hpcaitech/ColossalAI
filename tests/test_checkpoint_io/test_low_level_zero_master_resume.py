from copy import deepcopy

import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.testing import spawn


def _run_resume(
    rank, world_size, port, stage, offload, shard, low_cpu_mem_mode, checkpoint_dir, precision="bf16", cold=False
):
    torch.set_num_threads(1)
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost", seed=42)
    try:

        def create():
            model = torch.nn.Sequential(torch.nn.Linear(7, 11), torch.nn.GELU(), torch.nn.Linear(11, 5))
            if cold:
                model.register_parameter("unused", torch.nn.Parameter(torch.randn(13)))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
            booster = Booster(
                plugin=LowLevelZeroPlugin(
                    stage=stage,
                    precision=precision,
                    cpu_offload=offload,
                    overlap_communication=False,
                    master_weights=True,
                    max_norm=0,
                    initial_scale=32,
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

        if not cold:
            step(booster, model, optimizer)
        master_before = [p.detach().clone() for p in optimizer.working_to_master_param.values()]
        if not cold:
            dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            assert any(not torch.equal(p, p.to(dtype).float()) for p in master_before)
        model_path = f"{checkpoint_dir}/model" + ("" if shard else ".pt")
        optimizer_path = f"{checkpoint_dir}/optimizer" + ("" if shard else ".pt")
        booster.save_model(model, model_path, shard=shard)
        booster.save_optimizer(optimizer, optimizer_path, shard=shard, size_per_shard=0.001)
        dist.barrier()
        # Saving must not insert checkpoint metadata into the live Adam state.
        assert all("_colossalai_master_param" not in state for state in optimizer.optim.state.values())
        snapshot = optimizer.state_dict()
        saved_master = deepcopy([state["_colossalai_master_param"] for state in snapshot["state"].values()])
        expected_losses = [step(booster, model, optimizer) for _ in range(2)]
        for actual, expected in zip(snapshot["state"].values(), saved_master):
            torch.testing.assert_close(actual["_colossalai_master_param"], expected, atol=0, rtol=0)

        restored_booster, restored_model, restored_optimizer = create()
        restored_booster.load_model(restored_model, model_path, low_cpu_mem_mode=low_cpu_mem_mode)
        restored_booster.load_optimizer(restored_optimizer, optimizer_path, low_cpu_mem_mode=low_cpu_mem_mode)
        for actual, expected in zip(restored_optimizer.working_to_master_param.values(), master_before):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert all("_colossalai_master_param" not in state for state in restored_optimizer.optim.state.values())
        for expected in expected_losses:
            torch.testing.assert_close(
                step(restored_booster, restored_model, restored_optimizer), expected, atol=0, rtol=0
            )
        for actual, expected in zip(restored_model.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for actual, expected in zip(
            restored_optimizer.working_to_master_param.values(), optimizer.working_to_master_param.values()
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        # Legacy PyTorch-compatible states without master metadata still load.
        legacy = optimizer.state_dict()
        for state in legacy["state"].values():
            state.pop("_colossalai_master_param")
        restored_optimizer.load_state_dict(legacy)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("shard", [False, True])
@pytest.mark.parametrize("low_cpu_mem_mode", [False, True])
def test_low_level_zero_master_resume(world_size, stage, offload, shard, low_cpu_mem_mode, tmp_path):
    spawn(
        _run_resume,
        world_size,
        stage=stage,
        offload=offload,
        shard=shard,
        low_cpu_mem_mode=low_cpu_mem_mode,
        checkpoint_dir=str(tmp_path),
    )


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("shard", [False, True])
@pytest.mark.parametrize("precision,cold", [("fp16", False), ("fp16", True), ("bf16", True)])
def test_low_level_zero_master_resume_edge_cases(world_size, shard, precision, cold, tmp_path):
    spawn(
        _run_resume,
        world_size,
        stage=2,
        offload=True,
        shard=shard,
        low_cpu_mem_mode=True,
        checkpoint_dir=str(tmp_path),
        precision=precision,
        cold=cold,
    )
