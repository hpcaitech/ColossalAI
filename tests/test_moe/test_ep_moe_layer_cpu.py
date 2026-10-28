import pytest
import torch.distributed as dist

from colossalai.moe import EPMoELayer, ReferenceExperts
from colossalai.tensor.moe_tensor.api import ExpertGradScale, get_ep_group, get_ep_size, get_expert_grad_scale


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
