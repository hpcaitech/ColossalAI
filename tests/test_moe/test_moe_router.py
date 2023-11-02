import pytest
import torch

from colossalai.moe.routers import MoeRouter, Top1Router, Top2Router, TopKRouter


@pytest.mark.parametrize(["router", "num_groups"], [
    (Top1Router(), 1),
    (Top2Router(), 1),
    (TopKRouter(num_selected_experts=3), 4),
])
@pytest.mark.parametrize(["batch_size", "seq_len", "num_experts"], [
    (4, 5, 8),
    (3, 4, 4),
])
def test_router_forward(router: MoeRouter, batch_size: int, seq_len: int, num_experts: int, num_groups: int):
    x = torch.randn((batch_size * seq_len, num_experts)).cuda()
    if num_groups > 1:
        x = x.expand(num_groups, -1, -1)

    router.train()
    if isinstance(router, TopKRouter):
        combine_array, dispatch_mask = router(x, expert_capacity=2)
    else:
        combine_array, dispatch_mask = router(x)
    assert combine_array.shape[:-1] == x.shape
    assert dispatch_mask.shape[:-1] == x.shape
    assert torch.all(dispatch_mask.sum(-1).sum(-1) <= router.k_value)

    router.eval()
    if isinstance(router, TopKRouter):
        combine_array, dispatch_mask = router(x, expert_capacity=2)
    else:
        combine_array, dispatch_mask = router(x)
    assert combine_array.shape[:-1] == x.shape
    assert dispatch_mask.shape[:-1] == x.shape
    assert torch.all(dispatch_mask.sum(-1).sum(-1) <= router.k_value)


if __name__ == "__main__":
    test_router_forward(Top1Router(), 4, 4, 4, 1)
