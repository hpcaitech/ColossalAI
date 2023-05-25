from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.adam import Adam

from colossalai.nn.optimizer.fused_adam import FusedAdam


class FC(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(64, 64))

    def forward(self, x):
        return self.fc(x)


@pytest.mark.parametrize('adamw', [False, True])
@pytest.mark.parametrize('running_p_dtype', [torch.float, torch.half, torch.bfloat16])
@pytest.mark.parametrize('fp32_master_weights', [False, True])
def test_adam(adamw, running_p_dtype, fp32_master_weights):
    # baseline is fure fp32 torch adam
    # g_type is the same as running_p_dtype
    if running_p_dtype is torch.float and not fp32_master_weights:
        # pure fp32 must have fp32 weights
        return
    if not fp32_master_weights or running_p_dtype is torch.bfloat16:
        # pure low precision or bf16, high tolerance
        atol = 4e-3
        rtol = 4e-3
    else:
        # fp32 master weights, low tolerance
        atol = 2e-3
        rtol = 2e-3
    torch_model = FC().cuda()
    model = deepcopy(torch_model).to(running_p_dtype)

    torch_optim_cls = AdamW if adamw else Adam
    torch_optim = torch_optim_cls(torch_model.parameters(), lr=1e-3)
    optim = FusedAdam(model.parameters(), lr=1e-3, adamw_mode=adamw)

    data = torch.rand(10, 64).cuda()
    label = torch.rand(10, 64).cuda()

    for d, l in zip(data.to(running_p_dtype), label.to(running_p_dtype)):
        y = model(d)
        loss = ((l - y)**2).sum()
        optim.zero_grad()
        loss.backward()
        if fp32_master_weights:
            for p in model.parameters():
                p.data = p.data.float()
        optim.step()
        if fp32_master_weights:
            for p in model.parameters():
                p.data = p.data.to(running_p_dtype)

    for d, l in zip(data, label):
        y = torch_model(d)
        loss = ((l - y)**2).sum()
        torch_optim.zero_grad()
        loss.backward()
        torch_optim.step()

    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        if torch.isnan(p).any() or torch.isnan(torch_p).any():
            continue
        fp32_p = p.float()
        assert torch.allclose(fp32_p, torch_p, atol=atol, rtol=rtol)
