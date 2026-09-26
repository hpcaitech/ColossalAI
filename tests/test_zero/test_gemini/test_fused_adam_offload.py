import pytest
import torch
import torch.nn as nn

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.nn.optimizer import FusedAdam, HybridAdam
from colossalai.testing import rerun_if_address_is_in_use, spawn


class MLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=10, num_classes=10):
        super().__init__()
        layers = []
        for index in range(num_layers):
            in_dim = input_dim if index == 0 else hidden_dim
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.ReLU()))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_fused_adam_offload(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    torch.manual_seed(1024)
    model = MLP()
    optimizer = FusedAdam(model.parameters(), lr=1e-3)
    plugin = GeminiPlugin(offload_optim_frac=1.0, min_chunk_size_m=2)
    booster = Booster(plugin=plugin)
    model, optimizer, _, _, _ = booster.boost(model, optimizer)

    assert any(device.type == "cpu" for device in model.grads_device.values())
    assert type(optimizer.optim) is HybridAdam

    for _ in range(2):
        optimizer.zero_grad()
        output = model(torch.randn(4, 1024, device="cuda", dtype=torch.float16))
        booster.backward(output.float().square().mean(), optimizer)
        optimizer.step()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_fused_adam_offload():
    spawn(run_fused_adam_offload, 2)


if __name__ == "__main__":
    test_fused_adam_offload()
