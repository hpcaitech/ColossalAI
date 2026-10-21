from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch import nn

from colossalai.checkpoint_io.moe_checkpoint import MoECheckpointIO
from colossalai.interface.model import ModelWrapper
from colossalai.tensor.moe_tensor.api import set_moe_tensor_ep_fused, set_moe_tensor_ep_group
from colossalai.testing.utils import spawn

NUM_EXPERTS = 4
EP_SIZE = 2
NUM_LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE


class _PerExpertMoE(nn.Module):
    """Experts kept as separate modules, as in the transformers v4 Mixtral / DeepSeek EP blocks.

    Each EP rank keeps only its own experts; the others are released, so parameter names differ across ranks.
    """

    def __init__(self, ep_rank: int):
        super().__init__()
        self.experts = nn.ModuleList(nn.Linear(2, 2, bias=False) for _ in range(NUM_EXPERTS))
        held = range(ep_rank * NUM_LOCAL_EXPERTS, (ep_rank + 1) * NUM_LOCAL_EXPERTS)
        for idx, expert in enumerate(self.experts):
            if idx in held:
                expert.weight.data.fill_(float(idx))
            else:
                expert.weight = None


class _FusedMoE(nn.Module):
    """Experts stacked along dim 0 of one parameter, as in the transformers v5 Mixtral / DeepSeek-v3 EP blocks."""

    def __init__(self, ep_rank: int):
        super().__init__()
        start = ep_rank * NUM_LOCAL_EXPERTS
        self.gate_up_proj = nn.Parameter(_full_fused_weight()[start : start + NUM_LOCAL_EXPERTS].clone())


def _full_fused_weight() -> torch.Tensor:
    return torch.arange(NUM_EXPERTS * 2 * 2, dtype=torch.float32).reshape(NUM_EXPERTS, 2, 2)


def _collect(model: nn.Module) -> dict:
    state = {}
    for shard, _ in MoECheckpointIO._model_sharder(model):
        state.update(shard)
    return state


def _check_model_sharder(rank: int, world_size: int, port: int):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        ep_group = dist.new_group(ranks=list(range(world_size)))

        # Per-expert parameters are only tagged with the EP group: each rank keeps its own experts.
        per_expert = _PerExpertMoE(rank)
        for p in per_expert.parameters():
            set_moe_tensor_ep_group(p, ep_group)
        state = _collect(per_expert)
        held = range(rank * NUM_LOCAL_EXPERTS, (rank + 1) * NUM_LOCAL_EXPERTS)
        assert set(state) == {f"experts.{idx}.weight" for idx in held}
        for idx in held:
            assert torch.equal(state[f"experts.{idx}.weight"], torch.full((2, 2), float(idx)))

        # Fused parameters are gathered along the expert dimension to the EP root.
        fused = _FusedMoE(rank)
        set_moe_tensor_ep_group(fused.gate_up_proj, ep_group)
        set_moe_tensor_ep_fused(fused.gate_up_proj)
        state = _collect(fused)
        if rank == 0:
            assert torch.equal(state["gate_up_proj"], _full_fused_weight())
        else:
            assert torch.equal(state["gate_up_proj"], fused.gate_up_proj.detach())
    finally:
        dist.destroy_process_group()


def test_model_sharder_gathers_only_fused_experts():
    spawn(_check_model_sharder, nprocs=EP_SIZE)


@pytest.mark.parametrize("fused", [False, True])
def test_non_root_ep_rank_writes_only_per_expert_checkpoints(monkeypatch, tmp_path, fused):
    from colossalai.checkpoint_io import moe_checkpoint as moe_checkpoint_module

    module = _FusedMoE(ep_rank=1) if fused else _PerExpertMoE(ep_rank=1)
    for p in module.parameters():
        set_moe_tensor_ep_group(p, None)
        if fused:
            set_moe_tensor_ep_fused(p)

    checkpoint_io = object.__new__(MoECheckpointIO)
    checkpoint_io.moe_dp_rank = 0
    checkpoint_io.ep_rank = 1
    checkpoint_io.ep_size = EP_SIZE
    checkpoint_io.tp_rank = 0
    checkpoint_io.sp_rank = 0
    checkpoint_io.pp_rank = 0
    checkpoint_io.pp_size = 1
    checkpoint_io.verbose = False

    class _Coordinator:
        def is_master(self):
            return False

    checkpoint_io.coordinator = _Coordinator()
    observed = {}

    def fake_save_state_dict_shards(sharded_state_dict, checkpoint, index_file, base_filename, is_master, **kwargs):
        observed["is_master"] = is_master
        return 0

    monkeypatch.setattr(MoECheckpointIO, "_model_sharder", staticmethod(lambda model, **kwargs: iter(())))
    monkeypatch.setattr(moe_checkpoint_module, "save_state_dict_shards", fake_save_state_dict_shards)
    monkeypatch.setattr(moe_checkpoint_module.dist, "barrier", lambda: None)

    checkpoint_io.save_sharded_model(ModelWrapper(module), str(tmp_path / "checkpoint"))

    # Fused experts are gathered to the EP root, which writes them; per-expert parameters have
    # different names on each EP rank, so every EP rank writes its own experts.
    assert observed["is_master"] is (not fused)
