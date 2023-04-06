import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.adam import Adam

from colossalai.nn.optimizer.fused_adam import FusedAdam
from colossalai.testing import clear_cache_before_run, parameterize


class FC(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(64, 64))

    def forward(self, x):
        return self.fc(x)


@clear_cache_before_run()
@parameterize('adamw', [False, True])
@parameterize('p_dtype', [torch.float, torch.half])
@parameterize('g_dtype', [torch.float, torch.half])
def test_adam(adamw, p_dtype, g_dtype):
    model = FC().cuda().to(p_dtype)
    state = model.state_dict()
    model_copy = FC().cuda().to(p_dtype)
    model_copy.load_state_dict(state.copy())

    if adamw:
        optim = FusedAdam(model.parameters(), lr=1e-3, adamw_mode=True)
        torch_optim = AdamW(model_copy.parameters(), lr=1e-3)
    else:
        optim = FusedAdam(model.parameters(), lr=1e-3)
        torch_optim = Adam(model_copy.parameters(), lr=1e-3)

    data = torch.rand(1024, 64).cuda().to(p_dtype)
    data_copy = data.clone()
    label = torch.rand(1024, 64).cuda().to(p_dtype)

    for d, l in zip(data, label):
        y = model(d)
        loss = ((l - y)**2).sum()
        optim.zero_grad()
        loss.backward()
        if p_dtype != g_dtype:
            for i in range(len(optim.param_groups[0]['params'])):
                optim.param_groups[0]['params'][i].grad.data = optim.param_groups[0]['params'][i].grad.data.to(g_dtype)
        optim.step()

    for d, l in zip(data_copy, label):
        y = model_copy(d)
        loss = ((l - y)**2).sum()
        torch_optim.zero_grad()
        loss.backward()
        torch_optim.step()

    assert len(optim.param_groups[0]['params']) == len(torch_optim.param_groups[0]['params'])

    for i in range(len(optim.param_groups[0]['params'])):
        if torch.isnan(optim.param_groups[0]['params'][i]).any() \
           or torch.isnan(torch_optim.param_groups[0]['params'][i]).any():
            continue
        assert torch.allclose(optim.param_groups[0]['params'][i], torch_optim.param_groups[0]['params'][i], 2e-3, 2e-3)
