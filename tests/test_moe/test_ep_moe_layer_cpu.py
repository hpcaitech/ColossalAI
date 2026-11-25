import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.moe import EPMoELayer, ReferenceExperts
from colossalai.tensor.moe_tensor.api import (
    ExpertGradScale,
    get_ep_group,
    get_ep_size,
    get_expert_grad_scale,
    is_moe_tensor_ep_fused,
)
from colossalai.testing import rerun_if_address_is_in_use, spawn

NUM_EXPERTS, HIDDEN, INTERMEDIATE = 4, 8, 4


@pytest.mark.parametrize("initialized", [False, True])
def test_ep_moe_layer_default_process_group(initialized: bool, tmp_path):
    """Default-group expert tags work with the parameter API without changing local construction."""
    assert not dist.is_initialized()
    if initialized:
        dist.init_process_group("gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    try:
        layer = EPMoELayer(4, 8, 4, experts=ReferenceExperts(), device="cpu")
        expected_group = dist.group.WORLD if initialized else None
        assert layer.ep_group is expected_group
        assert layer.dispatcher.ep_group is expected_group
        assert layer.ep_size == 1
        assert layer.ep_rank == 0
        assert layer.num_local_experts == 4
        for param in layer.parameters():
            assert get_ep_group(param) is expected_group
            assert get_expert_grad_scale(param) == ExpertGradScale.IN_FORWARD
            if initialized:
                assert get_ep_size(param) == 1
    finally:
        if initialized:
            dist.destroy_process_group()


def _full_weights():
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, generator=generator),
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, generator=generator),
    )


def _init_gloo(rank, world_size, port):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    return dist.new_group(list(range(world_size)))


def _ep_layer(ep_group):
    return EPMoELayer(NUM_EXPERTS, HIDDEN, INTERMEDIATE, ep_group=ep_group, experts=ReferenceExperts(), device="cpu")


def test_ep_moe_layer_marks_experts_ep_fused():
    """``MoECheckpointIO`` gathers the experts of all EP ranks only for parameters marked ``ep_fused``."""
    layer = _ep_layer(None)
    assert all(is_moe_tensor_ep_fused(param) for param in layer.parameters())


def _run_load_full_experts(rank, world_size, port):
    ep_group = _init_gloo(rank, world_size, port)
    try:
        layer = _ep_layer(ep_group)
        gate_up, down = _full_weights()
        local = layer.local_expert_range
        local_gate_up, local_down = gate_up[local.start : local.stop], down[local.start : local.stop]

        # the full [num_experts, ...] tensors of a checkpoint: this rank keeps its own experts
        layer.load_state_dict({"gate_up_proj": gate_up, "down_proj": down})
        assert torch.equal(layer.gate_up_proj, local_gate_up) and torch.equal(layer.down_proj, local_down)

        # a tensor that already has the local shape is loaded as is
        layer.load_state_dict({"gate_up_proj": 2 * local_gate_up, "down_proj": 2 * local_down})
        assert torch.equal(layer.gate_up_proj, 2 * local_gate_up) and torch.equal(layer.down_proj, 2 * local_down)

        # any other shape is still reported by load_state_dict
        with pytest.raises(RuntimeError, match="size mismatch"):
            layer.load_state_dict({"gate_up_proj": gate_up[:3], "down_proj": local_down})
        with pytest.raises(RuntimeError, match="size mismatch"):
            layer.load_state_dict({"gate_up_proj": torch.tensor(1.0), "down_proj": local_down})
    finally:
        dist.destroy_process_group()


@rerun_if_address_is_in_use()
def test_ep_moe_layer_loads_full_experts():
    spawn(_run_load_full_experts, 2)


def _run_checkpoint_roundtrip(rank, world_size, port):
    from colossalai.checkpoint_io.moe_checkpoint import MoECheckpointIO

    ep_group = _init_gloo(rank, world_size, port)
    try:
        model = nn.Module()
        model.experts = _ep_layer(ep_group)
        gate_up, down = _full_weights()
        model.experts.load_full_weights(gate_up, down)

        saved = {}
        for shard, _ in MoECheckpointIO._model_sharder(model):
            saved.update(shard)
        if rank == 0:
            # the root rank writes the experts of all EP ranks, i.e. the full [num_experts, ...] tensors
            assert torch.equal(saved["experts.gate_up_proj"], gate_up)
            assert torch.equal(saved["experts.down_proj"], down)
        payload = [saved if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)

        # every rank reads the same files back, into a layer that starts from other weights
        restored = nn.Module()
        restored.experts = _ep_layer(ep_group)
        restored.load_state_dict(payload[0])
        local = restored.experts.local_expert_range
        assert torch.equal(restored.experts.gate_up_proj, gate_up[local.start : local.stop])
        assert torch.equal(restored.experts.down_proj, down[local.start : local.stop])
    finally:
        dist.destroy_process_group()


@rerun_if_address_is_in_use()
def test_ep_moe_layer_checkpoint_roundtrip():
    spawn(_run_checkpoint_roundtrip, 2)
