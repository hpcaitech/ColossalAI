import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.adam import Adam

from colossalai.nn.optimizer.hybrid_adam import HybridAdam
from colossalai.testing import clear_cache_before_run, parameterize

RE = 3


@clear_cache_before_run()
@parameterize('adamw', [False, True])
@parameterize('device', ['cpu', 'cuda:0'])
@parameterize('p_dtype', [torch.float])
@parameterize('g_dtype', [torch.float, torch.half])
def test_adam(adamw, device, p_dtype, g_dtype):
    rng_state = torch.get_rng_state()
    p = nn.Parameter(torch.rand(64).to(device, p_dtype))
    torch.set_rng_state(rng_state)
    p_copy = nn.Parameter(torch.rand(64).to(device).float())

    if adamw:
        optim = HybridAdam([p], lr=1e-3, adamw_mode=True)
        torch_optim = AdamW([p_copy], lr=1e-3)
    else:
        optim = HybridAdam([p], lr=1e-3)
        torch_optim = Adam([p_copy], lr=1e-3)

    print(f"adaw mode {adamw}, device {device}, p_dtype {p_dtype}, g_dtype {g_dtype}")
    for i in range(RE):
        p.grad = torch.rand(64).to(device, p_dtype)
        p_copy.grad = p.grad.clone().float()
        p.grad.data = p.grad.data.to(g_dtype)

        optim.step()
        torch_optim.step()

        if torch.isnan(p.data).any() or torch.isnan(p_copy.data).any():
            continue
        assert torch.allclose(p.data, p_copy.data, 1e-4, 1e-2), \
            f"adaw mode {adamw}, device {device}, p_dtype {p_dtype}, g_dtype {g_dtype}"
